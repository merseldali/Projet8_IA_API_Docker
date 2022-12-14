from fastapi import FastAPI, File
from starlette.responses import Response
import io
import zipfile
from segmentation import get_segmentation_model, get_segments, get_segments_by_id

model = get_segmentation_model()

app = FastAPI(title="DeepLabV3+ image segmentation",
              description='''Obtain semantic segmentation maps of the image in input via DeepLabV3+ implemented in 
              Keras. Visit this URL at port 8501 for the streamlit interface.''',
              version="0.1.0",
              )


@app.post("/segmentation")
def get_segmentation_map(file: bytes = File(...)):
    segmented_image = get_segments(model, file)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format='PNG')
    return Response(bytes_io.getvalue(), media_type="image/png")


@app.get("/segmentation/{image_id}")
def get_segmentation_map_by_id(image_id: int):
    raw_img, segmented_image, label_image = get_segments_by_id(model, image_id)
    if (raw_img is not None) and (segmented_image is not None) and (label_image is not None):
        bytes_io = io.BytesIO()
        bytes_io2 = io.BytesIO()
        bytes_io3 = io.BytesIO()
        raw_img.save(bytes_io, format='PNG')
        segmented_image.save(bytes_io2, format='PNG')
        label_image.save(bytes_io3, format='PNG')

        # Open StringIO to grab in-memory ZIP contents
        s = io.BytesIO()
        # The zip compressor
        zf = zipfile.ZipFile(s, "w")

        zf.writestr('raw.png', bytes_io.getvalue())
        zf.writestr('predicted.png', bytes_io2.getvalue())
        zf.writestr('labels.png', bytes_io3.getvalue())

        zf.close()

        return Response(s.getvalue(), media_type="application/x-zip-compressed")
    else:
        return Response()
