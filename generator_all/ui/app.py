import streamlit as st
from PIL import Image
import io
import zipfile
import os
import sys

# Add absolute path for src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
utils_path = os.path.join(current_dir, '..')

# Set HuggingFace cache - use D: drive locally, /root/.cache in containers
if os.path.exists('d:/'):
    os.environ['HF_HOME'] = 'd:/huggingface_cache'
    os.environ['TRANSFORMERS_CACHE'] = 'd:/huggingface_cache/transformers'
else:
    # Modal/container environment
    os.environ['HF_HOME'] = '/root/.cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface/transformers'

sys.path.insert(0, src_path)
sys.path.insert(0, utils_path)

st.set_page_config(
    page_title="Unified Data & Image Generator", 
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("Unified Data & Image Generator")

# Initialize S3 Manager
from utils.s3_storage import S3Manager
from utils.config import s3_config

# Validate S3 configuration
try:
    s3_config.validate()
    s3_manager = S3Manager()
    st.sidebar.success(f"✅ Connected to S3: {s3_config.bucket_name}")
except Exception as e:
    st.sidebar.error(f"⚠️ S3 Configuration Error: {str(e)}")
    st.sidebar.info("Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME environment variables")
    s3_manager = None

# Device selector (auto-detect)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device.upper()}")


# Cached model loaders (lazy initialization)
@st.cache_resource(show_spinner=False)
def get_caption_generator(_device):
    import torch
    from generators.caption_generator import CaptionGenerator
    dev = _device if _device == "cuda" and torch.cuda.is_available() else "cpu"
    try:
        return CaptionGenerator(dev)
    except (OSError, RuntimeError) as e:
        error_msg = str(e).lower()
        if "paging file" in error_msg or "memory" in error_msg:
            st.error("⚠️ Insufficient RAM to load the image captioning model. Try closing other applications.")
            return None
        elif "no space left" in error_msg or "errno 28" in error_msg:
            st.error("⚠️ Disk space full. Please free up space on D: drive (models need ~2GB).")
            return None
        raise

@st.cache_resource(show_spinner=False)
def get_variation_generator():
    from generators.variation_generator import VariationGenerator
    return VariationGenerator()

@st.cache_resource(show_spinner=False)
def get_image_generator(_device):
    import torch
    from generators.image_generator import ImageGenerator
    dev = _device if _device == "cuda" and torch.cuda.is_available() else "cpu"
    try:
        return ImageGenerator(dev)
    except (OSError, RuntimeError, MemoryError) as e:
        error_msg = str(e).lower()
        if "paging file" in error_msg or "memory" in error_msg:
            st.error("⚠️ Insufficient RAM to load the image generation model. Try closing other applications.")
            return None
        elif "no space left" in error_msg or "errno 28" in error_msg:
            st.error("⚠️ Disk space full. Please free up space on D: drive.")
            return None
        raise

@st.cache_resource(show_spinner=False)
def get_t5_model(_device):
    import torch
    from models.t5 import T5Model
    dev = _device if _device == "cuda" and torch.cuda.is_available() else "cpu"
    return T5Model(device=dev)

# Initialize session state for file uploader
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file...",
    type=['jpg', 'png', 'jpeg', 'csv', 'xlsx', 'docx', 'pdf']
)
st.divider()
custom_caption = st.text_area("Enter custom prompt (Only for image):", "")

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    st.success(f"File uploaded: {uploaded_file.name}")

    num_files = st.slider(
        "Number of output files to generate",
        min_value=1,
        max_value=100,
        value=10
    )

if st.button("Generate"):
    if uploaded_file is not None:
        if file_ext in ['jpg', 'png', 'jpeg']:
            # Image generation pipeline
            image = Image.open(uploaded_file).convert('RGB')

            # Optional: downscale large images to conserve memory
            max_dim = 1024
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim))

            if custom_caption.strip():
                # Use custom caption
                caption = custom_caption.strip()
                variations = [caption]
            else:
                # Generate caption from image
                with st.spinner("Generating caption..."):
                    import torch
                    with torch.no_grad():
                        caption_gen = get_caption_generator(device)
                        if caption_gen is None:
                            st.warning("Image generation skipped due to insufficient memory. Please try with tabular/text files instead.")
                            st.stop()
                        caption = caption_gen.generate_caption(image)

                # Adjust the number of caption variations to match num_files
                with st.spinner("Generating caption variations..."):
                    var_gen = get_variation_generator()
                    variations = var_gen.generate_variations(caption)
                    # Trim or extend variations to match num_files
                    if len(variations) < num_files:
                        variations = (variations * (num_files // len(variations) + 1))[:num_files]
                    else:
                        variations = variations[:num_files]

            st.subheader("Original Caption")
            st.write(caption)

            st.subheader("Caption Variations")
            for i, var in enumerate(variations, 1):
                st.write(f"{i}. {var}")

            with st.spinner("Generating images..."):
                import torch
                with torch.no_grad():
                    img_gen = get_image_generator(device)
                    num_images_to_generate = min(num_files, len(variations))
                    generated_images = img_gen.generate_images(variations[:num_images_to_generate])
            
            st.subheader("Generated Images")
            cols = st.columns(min(3, len(generated_images)))  # Show max 3 columns
            for i, img in enumerate(generated_images):
                with cols[i % 3]:
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
            
            # Upload images to S3 and provide download link
            if s3_manager:
                with st.spinner("Uploading images to S3..."):
                    try:
                        # Prepare image list for S3 upload
                        image_files = [(f"image_{i+1}.png", img) for i, img in enumerate(generated_images)]
                        
                        # Upload to S3 and get download URL
                        download_url = s3_manager.upload_images_and_zip(
                            images=image_files,
                            zip_name=f"generated_images_{uploaded_file.name.split('.')[0]}"
                        )
                        
                        st.success("✅ Images uploaded to S3!")
                        st.markdown(f"**[Download Images (ZIP)]({download_url})**")
                        st.info(f"⏰ Download link expires in {s3_config.presigned_url_expiration // 60} minutes")
                        
                    except Exception as e:
                        st.error(f"Failed to upload to S3: {str(e)}")
                        st.info("Falling back to local download...")
                        
                        # Fallback: Prepare ZIP for local download
                        img_zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(img_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for i, img in enumerate(generated_images):
                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                zip_file.writestr(f"image_{i+1}.png", img_byte_arr.getvalue())
                        
                        st.download_button(
                            label="Download Generated Images (ZIP)",
                            data=img_zip_buffer.getvalue(),
                            file_name="generated_images.zip",
                            mime="application/zip"
                        )
            else:
                # No S3 - use local download
                img_zip_buffer = io.BytesIO()
                with zipfile.ZipFile(img_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for i, img in enumerate(generated_images):
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        zip_file.writestr(f"image_{i+1}.png", img_byte_arr.getvalue())
                
                st.download_button(
                    label="Download Generated Images (ZIP)",
                    data=img_zip_buffer.getvalue(),
                    file_name="generated_images.zip",
                    mime="application/zip"
                )
        
        elif file_ext in ['csv', 'xlsx', 'docx', 'pdf']:
            with st.spinner('Generating output files...'):
                try:
                    result_data = None
                    if file_ext in ['csv', 'xlsx']:
                        from generators.tabular import TabularGenerator
                        generator = TabularGenerator()
                        result_data = generator.generate(uploaded_file, file_ext=file_ext, num_files=num_files)
                    elif file_ext == 'docx':
                        from generators.docx import DocxGenerator
                        t5_model = get_t5_model(device)
                        generator = DocxGenerator(t5_model)
                        result_data = generator.generate(uploaded_file, num_files=num_files)
                    elif file_ext == 'pdf':
                        from generators.pdf import PdfGenerator
                        t5_model = get_t5_model(device)
                        generator = PdfGenerator(t5_model)
                        result_data = generator.generate(uploaded_file, num_files=num_files)
                    
                    if result_data:
                        # Upload to S3 if available
                        if s3_manager:
                            with st.spinner("Uploading files to S3..."):
                                try:
                                    # Prepare files for S3 upload
                                    files_for_s3 = []
                                    for filename, data in result_data:
                                        # Determine content type based on file extension
                                        if file_ext == 'csv':
                                            content_type = 'text/csv'
                                        elif file_ext == 'xlsx':
                                            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                        elif file_ext == 'docx':
                                            content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                                        elif file_ext == 'pdf':
                                            content_type = 'application/pdf'
                                        else:
                                            content_type = 'application/octet-stream'
                                        
                                        files_for_s3.append((filename, data, content_type))
                                    
                                    # Upload to S3 and get download URL
                                    download_url = s3_manager.upload_and_zip(
                                        files=files_for_s3,
                                        zip_name=f"generated_{file_ext}_{uploaded_file.name.split('.')[0]}"
                                    )
                                    
                                    st.success("✅ Files uploaded to S3!")
                                    st.markdown(f"**[Download Generated Files (ZIP)]({download_url})**")
                                    st.info(f"⏰ Download link expires in {s3_config.presigned_url_expiration // 60} minutes")
                                    
                                except Exception as e:
                                    st.error(f"Failed to upload to S3: {str(e)}")
                                    st.info("Falling back to local download...")
                                    
                                    # Fallback: Prepare ZIP for local download
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                        for filename, data in result_data:
                                            zip_file.writestr(filename, data)
                                    
                                    st.download_button(
                                        label="Download Generated Files (ZIP)",
                                        data=zip_buffer.getvalue(),
                                        file_name="generated_data.zip",
                                        mime="application/zip"
                                    )
                        else:
                            # No S3 - use local download
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for filename, data in result_data:
                                    zip_file.writestr(filename, data)
                            
                            st.download_button(
                                label="Download Generated Files (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="generated_data.zip",
                                mime="application/zip"
                            )
                        
                        if file_ext in ['csv', 'xlsx']:
                            st.subheader("Preview of generated data (first file)")
                            first_filename, first_data = result_data[0]
                            if file_ext == 'csv':
                                import pandas as pd
                                st.dataframe(pd.read_csv(io.BytesIO(first_data)))
                            else:
                                import pandas as pd
                                st.dataframe(pd.read_excel(io.BytesIO(first_data)))
                
                except Exception as e:
                    st.error(f"Generation error: {e}")
        else:
            st.error("Unsupported file format")
    else:
        # Text-to-image generation when no file is uploaded
        if custom_caption.strip():
            prompt = custom_caption.strip()
            # Limit prompt length to avoid tokenization errors (Stable Diffusion max ~77 tokens, ~100 chars safe)
            if len(prompt) > 100:
                st.warning("Prompt is too long. Truncating to first 100 characters.")
                prompt = prompt[:100]
            with st.spinner("Generating image from text..."):
                import torch
                with torch.no_grad():
                    img_gen = get_image_generator(device)
                    if img_gen is None:
                        st.error("Failed to load image generation model.")
                        st.stop()
                    generated_images = img_gen.generate_images([prompt])

            st.subheader("Generated Image")
            st.image(generated_images[0], caption="Generated from text", use_container_width=True)

            # Download option
            if s3_manager:
                with st.spinner("Uploading image to S3..."):
                    try:
                        # Prepare image for S3 upload
                        image_files = [("generated_image.png", generated_images[0])]

                        # Upload to S3 and get download URL
                        download_url = s3_manager.upload_images_and_zip(
                            images=image_files,
                            zip_name="text_generated_image"
                        )

                        st.success("✅ Image uploaded to S3!")
                        st.markdown(f"**[Download Image]({download_url})**")
                        st.info(f"⏰ Download link expires in {s3_config.presigned_url_expiration // 60} minutes")

                    except Exception as e:
                        st.error(f"Failed to upload to S3: {str(e)}")
                        st.info("Falling back to local download...")

                        # Fallback: Local download
                        img_byte_arr = io.BytesIO()
                        generated_images[0].save(img_byte_arr, format='PNG')
                        st.download_button(
                            label="Download Generated Image",
                            data=img_byte_arr.getvalue(),
                            file_name="generated_image.png",
                            mime="image/png"
                        )
            else:
                # No S3 - local download
                img_byte_arr = io.BytesIO()
                generated_images[0].save(img_byte_arr, format='PNG')
                st.download_button(
                    label="Download Generated Image",
                    data=img_byte_arr.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )
        else:
            st.warning("Please enter a caption to generate an image from text.")

if __name__ == "__main__":
    pass  # Streamlit automatically calls this script
