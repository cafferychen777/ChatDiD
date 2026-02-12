#!/usr/bin/env python3
"""
Storage Manager for ChatDiD MCP Server

Handles file storage, resource management, and MCP resource serving
according to best practices.
"""

import os
import base64
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from urllib.request import url2pathname

logger = logging.getLogger(__name__)

# Single source of truth for extension → MIME type mapping.
_MIME_TYPES: Dict[str, str] = {
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.html': 'text/html',
    '.pdf': 'application/pdf',
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.tex': 'text/x-latex',
    '.md': 'text/markdown',
}


class StorageManager:
    """Manages file storage and MCP resource serving for ChatDiD."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize storage manager with robust path handling.
        
        Args:
            base_dir: Base directory for outputs. If None, uses environment 
                     variable or default.
        """
        # Configure base directory with multiple fallback strategies
        base_path = self._determine_safe_base_path(base_dir)
        
        # Initialize with the safest possible path
        self.base_dir = Path(base_path)
        
        # Ensure base directory is accessible and writable
        self._ensure_writable_directory()
        
        # Define subdirectories
        self.dirs = {
            'figures_png': self.base_dir / 'figures' / 'png',
            'figures_svg': self.base_dir / 'figures' / 'svg',
            'reports_html': self.base_dir / 'reports' / 'html',
            'reports_pdf': self.base_dir / 'reports' / 'pdf',
            'data_csv': self.base_dir / 'data' / 'csv',
            'data_json': self.base_dir / 'data' / 'json',
        }
        
        # Configuration
        self.max_files_per_type = int(os.getenv('CHATDID_MAX_FILES', '100'))
        self.cleanup_after_days = int(os.getenv('CHATDID_CLEANUP_DAYS', '30'))
        
        # Create directories
        self._initialize_directories()
    
    def _get_user_directories(self, home: Path) -> List[tuple]:
        """
        Get user-specific directories in a cross-platform way.

        Handles localized folder names on Windows/Linux and provides fallbacks.

        Args:
            home: User home directory path

        Returns:
            List of (dir_type, Path) tuples in priority order
        """
        import platform

        directories = []
        system = platform.system()

        # Try platformdirs if available (best option)
        try:
            import platformdirs
            user_docs = platformdirs.user_documents_dir()
            user_downloads = platformdirs.user_downloads_dir()
            directories.extend([
                ("user_documents_platformdirs", Path(user_docs)),
                ("user_downloads_platformdirs", Path(user_downloads)),
            ])
            logger.debug("Using platformdirs for user directories")
        except ImportError:
            logger.debug("platformdirs not available, using fallback methods")

        # Windows-specific handling
        if system == "Windows":
            try:
                # Try Windows API for proper localized paths
                import ctypes.wintypes
                from ctypes import windll

                CSIDL_PERSONAL = 5  # My Documents
                CSIDL_PROFILE = 40  # Downloads (Vista+)

                buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)

                # Get Documents folder
                windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, 0, buf)
                if buf.value:
                    directories.append(("user_documents_windows", Path(buf.value)))

                # Get user profile (for Downloads)
                windll.shell32.SHGetFolderPathW(None, CSIDL_PROFILE, None, 0, buf)
                if buf.value:
                    downloads = Path(buf.value) / "Downloads"
                    if downloads.exists():
                        directories.append(("user_downloads_windows", downloads))

                logger.debug("Using Windows API for user directories")
            except Exception as e:
                logger.debug(f"Windows API method failed: {e}")

        # Linux-specific handling (XDG user directories)
        elif system == "Linux":
            try:
                import subprocess
                # Try xdg-user-dir command for proper localized paths
                result = subprocess.run(
                    ["xdg-user-dir", "DOCUMENTS"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    documents = Path(result.stdout.strip())
                    if documents.exists():
                        directories.append(("user_documents_xdg", documents))

                result = subprocess.run(
                    ["xdg-user-dir", "DOWNLOAD"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    downloads = Path(result.stdout.strip())
                    if downloads.exists():
                        directories.append(("user_downloads_xdg", downloads))

                logger.debug("Using xdg-user-dir for user directories")
            except Exception as e:
                logger.debug(f"xdg-user-dir method failed: {e}")

        # Fallback: Try common English folder names
        common_docs = [
            "Documents",
            "My Documents",
            "文档",  # Chinese
            "ドキュメント",  # Japanese
            "Dokumente",  # German
            "Documentos",  # Spanish/Portuguese
        ]
        common_downloads = [
            "Downloads",
            "下载",  # Chinese
            "ダウンロード",  # Japanese
            "Descargas",  # Spanish
        ]

        for folder_name in common_docs:
            folder_path = home / folder_name
            if folder_path.exists() and folder_path.is_dir():
                directories.append(("user_documents_fallback", folder_path))
                break

        for folder_name in common_downloads:
            folder_path = home / folder_name
            if folder_path.exists() and folder_path.is_dir():
                directories.append(("user_downloads_fallback", folder_path))
                break

        # Always include plain home directory as final fallback
        directories.append(("user_home", home))

        return directories

    def _determine_safe_base_path(self, base_dir: Optional[str] = None) -> str:
        """
        Determine a safe, writable base path with multiple fallback strategies.
        
        Args:
            base_dir: Explicitly provided base directory
            
        Returns:
            Safe base directory path as string
        """
        # Priority order of path sources
        candidates = []
        
        # 1. Explicitly provided base_dir
        if base_dir:
            candidates.append(("explicit", base_dir))
        
        # 2. Environment variable (with safety checks)
        env_dir = os.getenv('CHATDID_OUTPUT_DIR')
        if env_dir:
            # Only use environment variable if it's not a dangerous system path
            if self._is_safe_path(env_dir):
                candidates.append(("environment", env_dir))
            else:
                logger.warning(f"Unsafe CHATDID_OUTPUT_DIR={env_dir}, ignoring")
        
        # 3. User HOME directory paths (BEST for Claude Desktop compatibility)
        user_home = Path.home()

        # Try to get proper user directories (cross-platform with fallbacks)
        user_dirs = self._get_user_directories(user_home)

        # Add user directory candidates
        for dir_type, dir_path in user_dirs:
            if dir_path:
                candidates.append((dir_type, dir_path / "ChatDiD_outputs"))

        # 4. Default relative paths (multiple fallbacks)
        candidates.extend([
            ("default", "./outputs"),
            ("fallback1", "outputs"),
            ("fallback2", "./chatdid_outputs"),
            ("fallback3", os.path.join(os.getcwd(), "outputs"))
        ])

        # Try each candidate until we find a workable one
        for source, path_str in candidates:
            try:
                path_obj = Path(path_str)

                # Test if we can create and write to this location
                if self._test_path_viability(path_obj):
                    logger.info(f"Using {source} base directory: {path_obj}")
                    return str(path_obj)
                else:
                    logger.debug(f"Path not viable ({source}): {path_obj}")

            except Exception as e:
                logger.debug(f"Error with path candidate {source} ({path_str}): {e}")

        # If all else fails, use user home directory as last resort (better than /tmp for Claude Desktop)
        home_fallback = user_home / "ChatDiD_temp_outputs"
        logger.warning(f"All path candidates failed, using home fallback: {home_fallback}")

        # Try to ensure this final fallback is at least writable
        try:
            home_fallback.mkdir(parents=True, exist_ok=True)
            test_file = home_fallback / '.chatdid_write_test'
            test_file.write_text('test', encoding='utf-8')
            test_file.unlink()
            logger.info(f"Final fallback directory verified: {home_fallback}")
        except Exception as e:
            logger.error(f"Even home fallback failed: {e}. Output functionality may not work.")

        return str(home_fallback)
    
    def _is_safe_path(self, path_str: str) -> bool:
        """
        Check if a path is safe to use (not a system directory).
        
        Args:
            path_str: Path to check
            
        Returns:
            True if path is safe to use
        """
        # Convert to Path object for analysis
        path = Path(path_str)
        
        # Dangerous absolute paths that should be avoided
        dangerous_paths = [
            '/outputs',  # Root system directory
            '/tmp/outputs',  # System temp (though less dangerous)
            '/var/outputs',  # System var directory
            '/usr/outputs',  # System usr directory
            '/etc/outputs',  # System config directory
        ]
        
        # Check if it's one of the known dangerous paths
        if str(path) in dangerous_paths:
            return False
        
        # Check if it starts with dangerous system roots (unless it's in user space)
        if path.is_absolute():
            # Allow paths under user directories
            user_home = Path.home()
            cwd = Path.cwd()
            
            # Safe if it's under user home or current working directory
            try:
                if user_home in path.parents or cwd in path.parents:
                    return True
            except (OSError, ValueError):
                # Handle edge cases in path resolution
                pass
            
            # Unsafe if it's directly under system roots
            dangerous_roots = ['/var', '/usr', '/etc', '/sys', '/proc']
            for root in dangerous_roots:
                if str(path).startswith(root + '/'):
                    return False
        
        return True  # Relative paths and safe absolute paths are OK
    
    def _test_path_viability(self, path: Path) -> bool:
        """
        Test if a path can be created and is writable.
        
        Args:
            path: Path to test
            
        Returns:
            True if path is viable
        """
        try:
            # Try to create the directory
            path.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = path / '.chatdid_write_test'
            test_file.write_text('test', encoding='utf-8')
            
            # Test read access
            content = test_file.read_text(encoding='utf-8')
            
            # Cleanup
            test_file.unlink()
            
            return content == 'test'
            
        except (PermissionError, OSError, IOError) as e:
            logger.debug(f"Path {path} not viable: {e}")
            return False
    
    def _ensure_writable_directory(self):
        """
        Ensure the base directory exists and is writable.
        
        If the current base_dir fails, try fallback strategies.
        """
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Test current base_dir
                if self._test_path_viability(self.base_dir):
                    logger.info(f"Confirmed writable directory: {self.base_dir}")
                    return
                
                # If failed, try fallback
                logger.warning(f"Directory {self.base_dir} not writable (attempt {attempt + 1})")
                
                if attempt == 0:
                    # Try ./outputs as fallback
                    fallback = Path("./outputs")
                elif attempt == 1:
                    # Try user home directory (better than /tmp for Claude Desktop)
                    fallback = Path.home() / "ChatDiD_outputs"
                else:
                    # Final fallback - current directory
                    fallback = Path(".") / "chatdid_temp_outputs"
                
                if self._test_path_viability(fallback):
                    logger.warning(f"Switching to fallback directory: {fallback}")
                    self.base_dir = fallback
                    return
                
                attempt += 1
                
            except Exception as e:
                logger.error(f"Error ensuring writable directory: {e}")
                attempt += 1
        
        # If we get here, all attempts failed
        raise RuntimeError(f"Cannot create writable output directory after {max_attempts} attempts")
    
    def _initialize_directories(self):
        """Create all required directories."""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized directory: {dir_path}")
    
    def get_output_path(self, file_type: str, method: str = "", 
                       extension: str = "png") -> Path:
        """
        Generate output path for a file.
        
        Args:
            file_type: Type of file (event_study, report, etc.)
            method: DID method used (optional)
            extension: File extension
            
        Returns:
            Full path for the output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine directory based on extension
        dir_map = {
            'png': self.dirs['figures_png'],
            'svg': self.dirs['figures_svg'],
            'html': self.dirs['reports_html'],
            'pdf': self.dirs['reports_pdf'],
            'csv': self.dirs['data_csv'],
            'json': self.dirs['data_json'],
        }
        
        output_dir = dir_map.get(extension.lower(), self.dirs['figures_png'])
        
        # Build filename
        if method:
            filename = f"{file_type}_{method}_{timestamp}.{extension}"
        else:
            filename = f"{file_type}_{timestamp}.{extension}"
        
        return output_dir / filename
    
    def save_file(self, content: Any, file_type: str, method: str = "",
                 extension: str = "png", custom_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save content to file and return metadata.

        Args:
            content: Content to save (bytes for binary, str for text)
            file_type: Type of file
            method: DID method used
            extension: File extension
            custom_path: Optional custom file path. If provided, saves to this location
                        instead of auto-generated path. Can be absolute or relative.

        Returns:
            Dictionary with file metadata and MCP resource info
        """
        # Use custom path if provided, otherwise generate standard path
        if custom_path is not None:
            file_path = Path(custom_path)
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path = self.get_output_path(file_type, method, extension)
        
        # Save based on content type
        if isinstance(content, bytes):
            with open(file_path, 'wb') as f:
                f.write(content)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Generate MCP resource URI (handles spaces, unicode, cross-platform)
        resource_uri = file_path.resolve().as_uri()
        
        # Get file stats
        file_stats = file_path.stat()
        
        return {
            'path': str(file_path.resolve()),
            'uri': resource_uri,
            'size': file_stats.st_size,
            'created': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            'type': file_type,
            'method': method,
            'extension': extension
        }
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """
        Get MCP resource for a file URI.
        
        Args:
            uri: File URI (e.g., file:///path/to/file.png)
            
        Returns:
            MCP resource dictionary
        """
        # Parse URI to get file path (handles percent-encoding and cross-platform).
        # url2pathname correctly strips the leading '/' on Windows drive letters
        # (e.g. '/C:/path' → 'C:\\path') while being a no-op on POSIX.
        parsed = urlparse(uri)
        if parsed.scheme != 'file':
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
        file_path = Path(url2pathname(parsed.path))
        
        if not file_path.exists():
            raise FileNotFoundError(f"Resource not found: {uri}")
        
        # Determine MIME type
        extension = file_path.suffix.lower()
        mime_type = _MIME_TYPES.get(extension, 'application/octet-stream')
        
        # Read content
        if extension in ['.png', '.pdf']:
            # Binary content - encode as base64
            with open(file_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            content_field = 'blob'
        else:
            # Text content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content_field = 'text'
        
        # Get file stats
        file_stats = file_path.stat()
        
        return {
            'uri': uri,
            'name': file_path.name,
            'title': self._format_title(file_path.name),
            'mimeType': mime_type,
            content_field: content,
            'size': file_stats.st_size,
            'annotations': {
                'audience': ['user'],
                'lastModified': datetime.fromtimestamp(
                    file_stats.st_mtime
                ).isoformat() + 'Z',
                'priority': 0.8  # Generated files are usually important
            }
        }
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """
        List all available resources.
        
        Returns:
            List of MCP resource metadata (without content)
        """
        resources = []
        
        for dir_name, dir_path in self.dirs.items():
            if not dir_path.exists():
                continue
            
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    # Determine MIME type
                    extension = file_path.suffix.lower()
                    mime_type = _MIME_TYPES.get(extension, 'application/octet-stream')
                    
                    # Get file stats
                    file_stats = file_path.stat()
                    
                    resources.append({
                        'uri': file_path.resolve().as_uri(),
                        'name': file_path.name,
                        'title': self._format_title(file_path.name),
                        'description': f"Generated {dir_name.replace('_', ' ')}",
                        'mimeType': mime_type,
                        'size': file_stats.st_size,
                        'annotations': {
                            'lastModified': datetime.fromtimestamp(
                                file_stats.st_mtime
                            ).isoformat() + 'Z'
                        }
                    })
        
        # Sort by modification time (newest first)
        resources.sort(
            key=lambda x: x['annotations']['lastModified'],
            reverse=True
        )
        
        return resources
    
    def cleanup_old_files(self) -> int:
        """
        Remove files older than retention period.
        
        Returns:
            Number of files removed
        """
        cutoff_time = datetime.now() - timedelta(days=self.cleanup_after_days)
        removed_count = 0
        
        for dir_path in self.dirs.values():
            if not dir_path.exists():
                continue
            
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    file_stats = file_path.stat()
                    file_time = datetime.fromtimestamp(file_stats.st_mtime)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        logger.info(f"Removed old file: {file_path}")
                        removed_count += 1
        
        return removed_count
    
    def enforce_storage_limits(self) -> int:
        """
        Enforce maximum file count per type.
        
        Returns:
            Number of files removed
        """
        removed_count = 0

        for _, dir_path in self.dirs.items():
            if not dir_path.exists():
                continue

            # Get all files with modification times
            files = []
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    file_stats = file_path.stat()
                    files.append((file_path, file_stats.st_mtime))
            
            # Check if over limit
            if len(files) > self.max_files_per_type:
                # Sort by modification time (oldest first)
                files.sort(key=lambda x: x[1])
                
                # Remove oldest files
                files_to_remove = len(files) - self.max_files_per_type
                for file_path, _ in files[:files_to_remove]:
                    file_path.unlink()
                    logger.info(f"Removed file (storage limit): {file_path}")
                    removed_count += 1
        
        return removed_count
    
    def _format_title(self, filename: str) -> str:
        """
        Format filename into readable title.
        
        Args:
            filename: Original filename
            
        Returns:
            Formatted title
        """
        # Remove extension
        name = Path(filename).stem
        
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        
        # Remove timestamp (assumes format _YYYYMMDD_HHMMSS at end)
        import re
        name = re.sub(r'\s+\d{8}\s+\d{6}$', '', name)
        
        # Capitalize words
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'base_directory': str(self.base_dir),
            'total_files': 0,
            'total_size': 0,
            'by_type': {}
        }
        
        for dir_name, dir_path in self.dirs.items():
            if not dir_path.exists():
                continue
            
            dir_stats = {
                'count': 0,
                'size': 0,
                'oldest': None,
                'newest': None
            }
            
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    file_stats = file_path.stat()
                    dir_stats['count'] += 1
                    dir_stats['size'] += file_stats.st_size
                    
                    file_time = datetime.fromtimestamp(file_stats.st_mtime)
                    if dir_stats['oldest'] is None or file_time < dir_stats['oldest']:
                        dir_stats['oldest'] = file_time
                    if dir_stats['newest'] is None or file_time > dir_stats['newest']:
                        dir_stats['newest'] = file_time
            
            if dir_stats['count'] > 0:
                dir_stats['oldest'] = dir_stats['oldest'].isoformat()
                dir_stats['newest'] = dir_stats['newest'].isoformat()
                stats['by_type'][dir_name] = dir_stats
                stats['total_files'] += dir_stats['count']
                stats['total_size'] += dir_stats['size']
        
        # Format total size
        stats['total_size_formatted'] = self._format_size(stats['total_size'])
        
        return stats
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    print(f"  Total size: {stats['total_size_formatted']}")