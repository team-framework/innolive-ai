import io
from PIL import Image

def to_grayscale(img_bytes):
    input_stream = io.BytesIO(img_bytes)
    img = Image.open(input_stream)

    bw_img = img.convert('L')

    output_stream = io.BytesIO()
    bw_img.save(output_stream, format=img.format or 'JPEG')
    bw_bytes = output_stream.getvalue()

    return bw_bytes