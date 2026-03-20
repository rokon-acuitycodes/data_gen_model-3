import io
import re
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader
from typing import List, Tuple, Any, IO
from generators.base import DataGenerator
from models.t5 import T5Model
from utils.helpers import get_paragraphs

class PdfGenerator(DataGenerator):
    """Generator for PDF documents."""

    def __init__(self, t5_model: T5Model):
        self.t5_model = t5_model

    def generate(self, original_file: IO[bytes], num_files: int = 100, **kwargs) -> List[Tuple[str, bytes]]:
        try:
            reader = PdfReader(original_file)
            num_pages = len(reader.pages)
            full_text = '\n\n'.join([page.extract_text() or '' for page in reader.pages])
            original_paras = get_paragraphs(full_text)
        except:
            num_pages = 1
            original_paras = ["Sample content for synthetic generation."]
            full_text = " ".join(original_paras)

        lines_per_page = 25

        generated_files = []

        for file_idx in range(num_files):
            # Use batch paraphrasing for faster processing
            synth_paras = self.t5_model.paraphrase_batch(original_paras, batch_size=5)
            synth_lines = []
            for para in synth_paras:
                wrapped_lines = textwrap.wrap(para, width=80)
                synth_lines.extend(wrapped_lines)
                synth_lines.append('')  # Add blank line for paragraph separation

            output_buffer = io.BytesIO()
            c = canvas.Canvas(output_buffer, pagesize=letter)
            width, height = letter
            line_idx = 0

            for i in range(num_pages):
                c.drawString(100, 750, f"Synthetic Page {i+1}")
                text_object = c.beginText(40, 700)

                page_end = min(line_idx + lines_per_page, len(synth_lines))
                for j in range(line_idx, page_end):
                    if synth_lines[j].strip():
                        text_object.textLine(synth_lines[j])
                line_idx = page_end

                c.drawText(text_object)
                c.showPage()

            c.save()
            generated_files.append((f"generated_{file_idx+1}_{original_file.name}", output_buffer.getvalue()))

        return generated_files
