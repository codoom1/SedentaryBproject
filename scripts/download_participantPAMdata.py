#!/usr/bin/env python3
"""
Python version of the downloadPartfiles R function.
Downloads and extracts participant logfiles from NHANES accelerometer data.
"""

import os
import sys
import tarfile
import tempfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from typing import Dict, Optional, Union
import re


def download_partfiles(
    participant_id: Union[int, str],
    cycle: str,
    dest_dir: Optional[str] = None,
    quiet: bool = True,
    delete_tar: bool = True,
    delete_extracted: bool = True
) -> Dict:
    """
    Download and extract participant logfiles from NHANES accelerometer data.
    
    Parameters
    ----------
    participant_id : int or str
        Participant SEQN identifier
    cycle : str
        Data cycle, either "2011-12" or "2013-14"
    dest_dir : str, optional
        Destination directory for downloads. If None, uses system temp directory
    quiet : bool, default True
        If True, suppress download progress messages
    delete_tar : bool, default True
        If True, delete the tar.bz2 archive after extraction
    delete_extracted : bool, default True
        If True, delete extracted files after processing
        
    Returns
    -------
    dict
        Dictionary containing:
        - id: participant ID
        - cycle: data cycle
        - url: download URL
        - archive_size_bytes: size of downloaded tar.bz2 file
        - logfile: name of logfile found in archive (or None)
        - logfile_size_bytes: size of logfile (or None)
        - logfile_header_only: whether logfile contains only header
        - destfile: path to tar file (or None if deleted)
        - error: error message if download failed
    """
    # Set up base URL based on cycle
    if cycle == "2011-12":
        base_url = "https://ftp.cdc.gov/pub/pax_g/"
    elif cycle == "2013-14":
        base_url = "https://ftp.cdc.gov/pub/pax_h/"
    else:
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': None,
            'error': f'Invalid cycle: {cycle}. Must be "2011-12" or "2013-14"'
        }
    
    url = f"{base_url}{participant_id}.tar.bz2"
    
    # Set up destination directory
    if dest_dir is None:
        dest_dir = tempfile.gettempdir()
    
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    tar_file = dest_path / f"{participant_id}.tar.bz2"
    
    # Download the file
    try:
        if not quiet:
            print(f"Downloading {url}...")

        # reporthook for urlretrieve to show progress
        def _reporthook(blocknum, blocksize, totalsize):
            if totalsize <= 0:
                # totalsize unknown
                downloaded = blocknum * blocksize
                sys.stdout.write(f"\rDownloaded {downloaded} bytes")
            else:
                downloaded = blocknum * blocksize
                percent = downloaded / totalsize * 100
                if percent > 100:
                    percent = 100
                sys.stdout.write(f"\rDownloading: {percent:5.1f}% ({downloaded}/{totalsize} bytes)")
            sys.stdout.flush()

        urlretrieve(url, tar_file, reporthook=_reporthook)
        if not quiet:
            print()  # newline after progress
    except Exception as e:
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': url,
            'error': f'download failed: {str(e)}'
        }
    
    # Check if file was downloaded successfully
    if not tar_file.exists():
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': url,
            'error': 'download failed: file not found after download'
        }
    
    archive_size = tar_file.stat().st_size
    
    # List contents of tar file
    try:
        with tarfile.open(tar_file, 'r:bz2') as tar:
            files = tar.getnames()
    except Exception as e:
        if delete_tar and tar_file.exists():
            tar_file.unlink()
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': url,
            'archive_size_bytes': archive_size,
            'logfile': None,
            'logfile_size_bytes': None,
            'logfile_header_only': None
        }
    
    if not files:
        if delete_tar and tar_file.exists():
            tar_file.unlink()
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': url,
            'archive_size_bytes': archive_size,
            'logfile': None,
            'logfile_size_bytes': None,
            'logfile_header_only': None
        }
    
    # Find logfile
    # First try to find files matching log.*(txt|log) pattern
    log_pattern = re.compile(r'log.*\.(txt|log)$', re.IGNORECASE)
    logfiles = [f for f in files if log_pattern.search(f)]
    
    # If no matches, try files containing 'log'
    if not logfiles:
        logfiles = [f for f in files if 'log' in f.lower()]
    
    logfile_name = logfiles[0] if logfiles else None
    
    # Extract and analyze logfile
    log_size = None
    header_only = None
    extract_dir = dest_path / f"ex_{participant_id}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if logfile_name:
        try:
            with tarfile.open(tar_file, 'r:bz2') as tar:
                tar.extract(logfile_name, extract_dir)
            
            logfile_path = extract_dir / logfile_name
            
            if logfile_path.exists():
                log_size = logfile_path.stat().st_size
                
                # Read first 2000 lines to check if header only
                try:
                    with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= 2000:
                                break
                            lines.append(line)
                    
                    # Filter out empty lines and comments
                    non_empty_lines = [
                        line.strip() for line in lines 
                        if line.strip() and not re.match(r'^\s*[#;]', line)
                    ]
                    
                    # If only 1 or fewer non-empty, non-comment lines, it's header only
                    header_only = len(non_empty_lines) <= 1
                    
                    # If header only, set size to None
                    if header_only:
                        log_size = None
                        
                except Exception as e:
                    if not quiet:
                        print(f"Warning: Could not read logfile: {e}")
        except Exception as e:
            if not quiet:
                print(f"Warning: Could not extract logfile: {e}")
    
    # Cleanup
    if delete_tar and tar_file.exists():
        tar_file.unlink()
    
    if delete_extracted and extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    
    return {
        'id': participant_id,
        'cycle': cycle,
        'url': url,
        'archive_size_bytes': archive_size,
        'logfile': logfile_name,
        'logfile_size_bytes': log_size,
        'logfile_header_only': header_only,
        'destfile': str(tar_file) if not delete_tar else None
    }


def batch_download_logs(
    ids_11_12: list,
    ids_13_14: list,
    dest_dir: Optional[str] = None,
    per_file_progress: bool = True,
    out_csv: Optional[str] = None,
    delete_tar: bool = True,
    delete_extracted: bool = True
) -> list:
    """
    Batch download logfiles for multiple participants across cycles.
    
    Parameters
    ----------
    ids_11_12 : list
        List of participant IDs from 2011-12 cycle
    ids_13_14 : list
        List of participant IDs from 2013-14 cycle
    dest_dir : str, optional
        Destination directory for downloads
    per_file_progress : bool, default True
        If True, print progress for each file
    out_csv : str, optional
        Path to save results CSV file
    delete_tar : bool, default True
        If True, delete tar.bz2 archives after extraction
    delete_extracted : bool, default True
        If True, delete extracted files after processing
        
    Returns
    -------
    list
        List of dictionaries containing results for each participant
    """
    # Combine participant lists
    combos = []
    for pid in ids_11_12:
        combos.append({'id': int(pid), 'cycle': '2011-12'})
    for pid in ids_13_14:
        combos.append({'id': int(pid), 'cycle': '2013-14'})
    
    n = len(combos)
    results = []
    
    for i, combo in enumerate(combos, 1):
        pid = combo['id']
        cycle = combo['cycle']
        
        if per_file_progress:
            print(f"[{i:3d}/{n:3d}] SEQN {pid} ({cycle})")
        
        result = download_partfiles(
            pid, str(cycle),
            dest_dir=dest_dir,
            quiet=not per_file_progress,
            delete_tar=delete_tar,
            delete_extracted=delete_extracted
        )
        
        # Determine status
        if 'error' in result and result['error']:
            status = 'download_failed'
        elif result['logfile_header_only'] is None:
            status = 'unknown'
        elif result['logfile_header_only']:
            status = 'header_only'
        else:
            status = 'has_data'
        
        results.append({
            'id': result['id'],
            'cycle': result['cycle'],
            'logfile_header_only': result['logfile_header_only'],
            'logfile_size_bytes': result['logfile_size_bytes'],
            'status': status
        })
        
        # Simple progress bar
        if not per_file_progress:
            progress = int(50 * i / n)
            sys.stdout.write(f"\r[{'=' * progress}{' ' * (50 - progress)}] {i}/{n}")
            sys.stdout.flush()
    
    if not per_file_progress:
        print()  # New line after progress bar
    
    # Save to CSV if requested
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd  # type: ignore
            df = pd.DataFrame(results)  # type: ignore
            df.to_csv(out_csv, index=False)  # type: ignore
        except Exception:
            # Fallback to CSV writer if pandas isn't available
            import csv as _csv
            with out_path.open('w', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
                if results:
                    writer.writeheader()
                    writer.writerows(results)
        print(f"Results saved to: {out_csv}")
    
    return results

def download_participant_archive_only(
    participant_id: Union[int, str],
    cycle: str,
    output_dir: Optional[str] = None,
    quiet: bool = True,
    extract: bool = False,
    remove_archive_after_extract: bool = False
) -> Dict:
    """
    Download participant tar.bz2 archive and optionally extract files, without processing the logfile.

    Parameters
    ----------
    participant_id : int or str
        Participant SEQN identifier
    cycle : str
        Data cycle, either "2011-12" or "2013-14"
    output_dir : str, optional
        Directory to store the downloaded archive (and extracted files if extract=True).
        If None, uses the system temporary directory.
    quiet : bool, default True
        If False, prints progress messages.
    extract : bool, default False
        If True, extracts the full archive into a subdirectory named after the participant.
    remove_archive_after_extract : bool, default False
        If True and extract=True, removes the archive after successful extraction.

    Returns
    -------
    dict
        Dictionary containing keys: id, cycle, url, archive_path, archive_size_bytes,
        extracted_dir (or None), error (or None)
    """
    # Validate cycle and build base url (reuse logic from download_partfiles)
    if cycle == "2011-12":
        base_url = "https://ftp.cdc.gov/pub/pax_g/"
    elif cycle == "2013-14":
        base_url = "https://ftp.cdc.gov/pub/pax_h/"
    else:
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': None,
            'archive_path': None,
            'archive_size_bytes': None,
            'extracted_dir': None,
            'error': f'Invalid cycle: {cycle}. Must be "2011-12" or "2013-14"'
        }

    url = f"{base_url}{participant_id}.tar.bz2"

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    archive_path = out_path / f"{participant_id}.tar.bz2"

    try:
        if not quiet:
            print(f"Downloading {url} to {archive_path} ...")

        # Progress hook for urlretrieve
        def _format_bytes(num: int) -> str:
            size = float(num)
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024.0:
                    return f"{size:3.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} PB"

        def _reporthook(blocknum, blocksize, totalsize):
            downloaded = blocknum * blocksize
            if totalsize > 0:
                pct = min(100.0, downloaded * 100.0 / totalsize)
                width = 40
                filled = int(width * pct / 100.0)
                bar = '=' * filled + '>' + ' ' * max(0, width - filled - 1)
                sys.stdout.write(
                    f"\r[{bar}] {pct:5.1f}%  {_format_bytes(downloaded)}/{_format_bytes(totalsize)}"
                )
            else:
                sys.stdout.write(f"\rDownloaded {_format_bytes(downloaded)}")
            sys.stdout.flush()

        urlretrieve(url, archive_path, reporthook=_reporthook)
        if not quiet:
            print()  # newline after progress line
    except Exception as e:
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': url,
            'archive_path': None,
            'archive_size_bytes': None,
            'extracted_dir': None,
            'error': f'download failed: {str(e)}'
        }

    if not archive_path.exists():
        return {
            'id': participant_id,
            'cycle': cycle,
            'url': url,
            'archive_path': None,
            'archive_size_bytes': None,
            'extracted_dir': None,
            'error': 'download failed: file not found after download'
        }

    archive_size = archive_path.stat().st_size
    extracted_dir = None

    if extract:
        try:
            extracted_dir = out_path / f"{participant_id}"
            extracted_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, 'r:bz2') as tar:
                members = tar.getmembers()
                total_members = len(members)
                total_bytes = sum(m.size for m in members if m.isfile())

                extracted_members = 0
                extracted_bytes = 0

                def _format_bytes(num: int) -> str:
                    size = float(num)
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if size < 1024.0:
                            return f"{size:3.1f} {unit}"
                        size /= 1024.0
                    return f"{size:.1f} PB"

                for m in members:
                    tar.extract(m, path=extracted_dir)
                    extracted_members += 1
                    if m.isfile():
                        extracted_bytes += m.size
                    if not quiet:
                        if total_bytes > 0:
                            pct = min(100.0, extracted_bytes * 100.0 / total_bytes)
                            bar_w = 40
                            filled = int(bar_w * pct / 100.0)
                            bar = '=' * filled + '>' + ' ' * max(0, bar_w - filled - 1)
                            sys.stdout.write(
                                f"\rExtracting [{bar}] {pct:5.1f}%  {_format_bytes(extracted_bytes)}/{_format_bytes(total_bytes)}"
                            )
                        else:
                            pct = extracted_members * 100.0 / max(1, total_members)
                            sys.stdout.write(f"\rExtracting {pct:5.1f}%  {extracted_members}/{total_members} files")
                        sys.stdout.flush()
                if not quiet:
                    print()  # newline after progress bar

            if remove_archive_after_extract and archive_path.exists():
                archive_path.unlink()
                archive_path_val = None
            else:
                archive_path_val = str(archive_path)

        except Exception as e:
            return {
                'id': participant_id,
                'cycle': cycle,
                'url': url,
                'archive_path': str(archive_path),
                'archive_size_bytes': archive_size,
                'extracted_dir': None,
                'error': f'extraction failed: {str(e)}'
            }
    else:
        archive_path_val = str(archive_path)

    return {
        'id': participant_id,
        'cycle': cycle,
        'url': url,
        'archive_path': archive_path_val,
        'archive_size_bytes': archive_size,
        'extracted_dir': str(extracted_dir) if extracted_dir is not None else None,
        'error': None
    }



if __name__ == "__main__":
    # Example usage: download archive only (no logfile processing)
    if len(sys.argv) > 2:
        participant_id = sys.argv[1]
        cycle = sys.argv[2]

        # optional third argument: output directory
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None

        # optional flags: --extract and --remove-archive
        extract = '--extract' in sys.argv
        remove_archive_after_extract = '--remove-archive' in sys.argv

        result = download_participant_archive_only(
            participant_id,
            cycle,
            output_dir=output_dir,
            quiet=False,
            extract=extract,
            remove_archive_after_extract=remove_archive_after_extract
        )

        print("\nResult:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("Usage: python download_participantPAMdata.py <participant_id> <cycle> [output_dir] [--extract] [--remove-archive]")
        print('Example: python download_participantPAMdata.py 62161 "2011-12" data/raw --extract --remove-archive')
        print("\nOr import and use the functions in your own script:")
        print("  from download_participantPAMdata import download_participant_archive_only, download_partfiles, batch_download_logs")


