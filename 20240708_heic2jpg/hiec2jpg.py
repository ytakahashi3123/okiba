import os
from PIL import Image
import pillow_heif

def convert_heic_to_jpg(directory, resize_factor=1.0):
    # ディレクトリ内のすべてのファイルを確認
    for filename in os.listdir(directory):
        if filename.lower().endswith('.heic'):
            heic_path = os.path.join(directory, filename)
            jpg_path = os.path.join(directory, os.path.splitext(filename)[0] + '.jpg')
            
            # HEICファイルを読み込む
            heif_file = pillow_heif.read_heif(heic_path)
            
            # PILのイメージオブジェクトに変換
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )

            # 画像サイズを縮小
            if resize_factor != 1.0:
                new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # JPG形式で保存
            image.save(jpg_path, "JPEG")
            print(f"Converted {heic_path} to {jpg_path}")

# 使用例
directory = "./rera/"  # 変換したいディレクトリのパスに置き換えてください
resize_factor = 0.75  # 縮小率 (0.5は50%に縮小)
convert_heic_to_jpg(directory, resize_factor)
