from fastapi import FastAPI, File
from starlette.responses import Response
import io
import zipfile
from segmentation import get_segmentator, get_segments, get_segments_by_id

model = get_segmentator()

app = FastAPI(title="DeepLabV3+ image segmentation",
    description='''Obtain semantic segmentation maps of the image in input via DeepLabV3+ implemented in Keras. Visit this URL at port 8501 for the streamlit interface.''',
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
    segmented_image, label_image = get_segments_by_id(model, image_id)
    if (segmented_image != None) and (label_image != None):
        bytes_io = io.BytesIO()
        bytes_io2 = io.BytesIO()
        segmented_image.save(bytes_io, format='PNG')
        label_image.save(bytes_io2, format='PNG')
    
        # Open StringIO to grab in-memory ZIP contents
        #s = io.StringIO()
        s = io.BytesIO()
        # The zip compressor
        zf = zipfile.ZipFile(s, "w")
    
        zf.write('predicted.png', bytes_io.getvalue())
        zf.write('labels.png', bytes_io2.getvalue())
    
        zf.close()
    
        return Response(s.getvalue(), media_type="application/x-zip-compressed")
    else:
        return Response()