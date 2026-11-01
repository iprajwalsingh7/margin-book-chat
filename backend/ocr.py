"""Offline PDF rasterization and OCR using bundled ONNX models."""
import math
from pathlib import Path

MAX_OCR_PAGES = 500
LOW_CONFIDENCE = 0.85


def has_raster_image(page):
    """Find raster images, including images nested in Form XObjects."""
    def inspect(resources, depth=0):
        if not resources or depth > 8:
            return False
        resources = resources.get_object()
        objects = resources.get('/XObject', {})
        if hasattr(objects, 'get_object'):
            objects = objects.get_object()
        for reference in objects.values():
            obj = reference.get_object()
            if obj.get('/Subtype') == '/Image':
                return True
            if obj.get('/Subtype') == '/Form' and inspect(obj.get('/Resources'), depth + 1):
                return True
        return False
    return inspect(page.get('/Resources'))


class PdfOCR:
    def __init__(self, data):
        import pypdfium2 as pdfium
        self.document = pdfium.PdfDocument(data)
        self.engine = None

    def read_page(self, index):
        if self.engine is None:
            import rapidocr
            from rapidocr import RapidOCR
            models = Path(rapidocr.__file__).parent / 'models'
            paths = {
                'Det.model_path': models / 'PP-OCRv6_det_small.onnx',
                'Cls.model_path': models / 'ch_ppocr_mobile_v2.0_cls_mobile.onnx',
                'Rec.model_path': models / 'PP-OCRv6_rec_small.onnx',
            }
            if not all(path.is_file() for path in paths.values()):
                raise RuntimeError('Bundled OCR models are missing. Run uv sync --locked to restore them.')
            # Explicit local models prevent automatic model downloads during extraction.
            self.engine = RapidOCR(params={
                **{key: str(path) for key, path in paths.items()},
                'EngineConfig.onnxruntime.intra_op_num_threads': 2,
                'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                'Global.log_level': 'warning',
            })
        page = self.document[index]
        try:
            width, height = page.get_size()
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise ValueError('PDF page has invalid dimensions.')
            scale = min(200 / 72, 4000 / max(width, height), math.sqrt(12_000_000 / (width * height)))
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert('RGB')
                try:
                    result = self.engine(image)
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            page.close()
        lines = result.txts if result.txts is not None else ()
        scores = result.scores if result.scores is not None else ()
        text = '\n'.join(lines).strip()
        confidence = round(sum(float(score) for score in scores) / len(scores), 4) if scores else None
        return text, confidence

    def close(self):
        self.document.close()
