import urllib.request 
import urllib.error

# URL from the Azure Public Dataset GitHub repo (Section 6 of plan.md)
URL = "https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz"


def check_size():
    print(f"Checking size of: \n{URL}\n")
    
    # The 'Range' header asks for only the first byte. 
    # The server responds with 'Content-Range: bytes 0-0/TOTAL_SIZE', revealing the total size.
    req = urllib.request.Request(URL, headers={'Range': 'bytes=0-0'})
    
    try:
        with urllib.request.urlopen(req, timeout = 15) as response:
            content_range = response.headers.get('Content-Range')
            if content_range and '/' in content_range:
                size_bytes = int(content_range.split('/')[-1])
                size_gb = size_bytes / (1024**3)
                
                print(f"✅ Archive size: {size_bytes:,} bytes ({size_gb:.2f} GB)")
                print("\nNOTE: The uncompressed data will be significantly larger (likely 3x-5x).")
                print("Ensure you have enough free disk space before we proceed to download.")
            else:
                print("Couldn't not determine size from Content-Range header")
                print(f"Headers received: {dict(response.headers)}")
                
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}: {e.reason}")
        print("THE URL might have changed or requires authentication. Check the Azure Public Github Data")
    except Exception as e:
        print(f"Failed to check size: {e}")
        
if __name__ == "__main__":
    check_size()