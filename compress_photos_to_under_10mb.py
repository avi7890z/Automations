from PIL import Image
import os


def compress_photo(input_path, output_path=None, max_size_mb=10):
    """
    Compress a photo to under the specified size while maintaining quality.

    Args:
        input_path: Path to input image
        output_path: Path for output image (optional, defaults to input_compressed.jpg)
        max_size_mb: Maximum file size in MB (default 10)
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    # Set output path
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_compressed{ext}"

    # Open image
    img = Image.open(input_path)

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Check if image is already under size
    img.save(output_path, 'JPEG', quality=95, optimize=True)
    if os.path.getsize(output_path) <= max_size_bytes:
        print(f"✓ Image already under {max_size_mb}MB. Saved with quality=95")
        return output_path

    # Binary search for optimal quality
    low, high = 10, 95
    best_quality = low

    while low <= high:
        mid = (low + high) // 2
        img.save(output_path, 'JPEG', quality=mid, optimize=True)
        size = os.path.getsize(output_path)

        if size <= max_size_bytes:
            best_quality = mid
            low = mid + 1
        else:
            high = mid - 1

    # Save with best quality found
    img.save(output_path, 'JPEG', quality=best_quality, optimize=True)
    final_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"✓ Compressed to {final_size:.2f}MB with quality={best_quality}")
    return output_path


def compress_folder(folder_path, output_folder=None, max_size_mb=10):
    """
    Compress all images in a folder.

    Args:
        folder_path: Path to folder containing images
        output_folder: Output folder (optional, defaults to folder_path/compressed)
        max_size_mb: Maximum file size in MB
    """
    if output_folder is None:
        output_folder = os.path.join(folder_path, 'compressed')

    os.makedirs(output_folder, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder,
                                       os.path.splitext(filename)[0] + '.jpg')

            print(f"\nProcessing: {filename}")
            try:
                compress_photo(input_path, output_path, max_size_mb)
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")


# Example usage
if __name__ == "__main__":
    # Compress a single photo
    # compress_photo('photo.jpg', 'photo_compressed.jpg', max_size_mb=10)

    # Compress all photos in a folder
    folder_path = r'D:\Photos printouts\Combined'
    output_path = r'F:\Photos_printout_10mb'
    compress_folder(folder_path, output_path, max_size_mb=10)
