import streamlit as st
import numpy as np
import os
import matplotlib

from diffusion_mnist import DiffMNISTGenerator
matplotlib.use('Agg')  # Use non-interactive backend

st.set_page_config(
    page_title="Diffusion MNIST Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Diffusion MNIST Generator")
st.markdown("Generate MNIST digits using different techniques for diffusion models. Select a model and digit to create an animation of the generation process.")

# Sidebar for model and digit selection
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_options = ["Dummy", "DDPM", "DDIM"]
    selected_model = st.selectbox(
        "Select Diffusion Model",
        model_options,
        help="Choose which diffusion model architecture to use"
    )
    
    # Digit selection with a numeric keyboard
    st.subheader("Select Digit")
    col1, col2, col3 = st.columns(3)
    selected_digit = 0
    
    with col1:
        if st.button("1", use_container_width=True):
            selected_digit = 1
        if st.button("4", use_container_width=True):
            selected_digit = 4
        if st.button("7", use_container_width=True):
            selected_digit = 7
    
    with col2:
        if st.button("2", use_container_width=True):
            selected_digit = 2
        if st.button("5", use_container_width=True):
            selected_digit = 5
        if st.button("8", use_container_width=True):
            selected_digit = 8
        if st.button("0", use_container_width=True):
            selected_digit = 0
    
    with col3:
        if st.button("3", use_container_width=True):
            selected_digit = 3
        if st.button("6", use_container_width=True):
            selected_digit = 6
        if st.button("9", use_container_width=True):
            selected_digit = 9
    
    # Alternatively, use a number input
    # selected_digit = st.number_input(
    #     "Or enter digit (0-9)",
    #     min_value=0,
    #     max_value=9,
    #     value=0,
    #     step=1
    # )
    
    # Generation parameters
    # st.subheader("Generation Parameters")
    # num_timesteps = st.slider(
    #     "Number of timesteps",
    #     min_value=10,
    #     max_value=1000,
    #     value=50,
    #     help="More timesteps = better quality but slower generation"
    # )
    
    # seed control
    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Generation Preview")
    
    # Placeholder for generated image
    placeholder = st.empty()
    placeholder.image(np.zeros((28, 28)), caption="Generated digit will appear here", width='stretch')
    
    generate_button = st.button(
        "🚀 Generate Digit",
        type="primary",
        use_container_width=True
    )

with col2:
    st.subheader("Model Status")
    
    # Check if weights exist
    weights_exist = os.path.exists(f"./weights/{selected_model.lower()}.pt")
    
    if weights_exist:
        st.success(f"✅ {selected_model} weights loaded")
    else:
        st.warning(f"⚠️ {selected_model} weights not found")
        st.info("Place model weights in ./weights/ directory")
    
    # Display selected options
    st.metric("Selected Model", selected_model)
    st.metric("Selected Digit", selected_digit)

# When generate button is clicked
if generate_button:
    with st.spinner(f"Generating digit {selected_digit} using {selected_model}..."):
        # Initialize model and diffusion process
        model = DiffMNISTGenerator(selected_model, seed)

        try:
            (final_img, timestamps) = model.generate(selected_digit)

        except Exception as e:
            st.error(f"Error during generation: {str(e)}")

        # # Update session state
        # st.session_state["last_gif"] = gif_path
        # st.session_state["timestamp"] = timestamp + 1
        
        # Display the final image
        placeholder.image(final_img, caption=f"Generated digit: {selected_digit}", width='stretch')
        
        # Display success message
        # st.success(f"✅ Generation complete! GIF saved to {gif_path}")
        
        # # Show the generated GIF
        # st.subheader("Generation Animation")
        # st.image(gif_path, caption="Diffusion process animation", width='content')
        
        # Provide download button
        # with open(gif_path, "rb") as file:
        #     st.download_button(
        #         label="📥 Download GIF",
        #         data=file,
        #         file_name=f"digit_{selected_digit}_{selected_model}.gif",
        #         mime="image/gif",
        #         use_container_width=True
        #     )
            
        
# Instructions section
st.divider()
with st.expander("📋 Instructions"):
    st.markdown("""
    ### How to use this application:
    1. **Select a model** from the dropdown in the sidebar
    2. **Choose a digit** (0-9) using the numeric keyboard or input field
    3. **Adjust parameters** if needed (number of timesteps, seed)
    4. **Click 'Generate Digit'** to start the diffusion process
    5. **Wait for the animation** to complete and download the GIF
    
    ### Model weights:
    - Place your trained model weights in the `./weights/` directory
    - Name them according to the model type (e.g., `ddpm.pth`, `ddim.pth`)
    - The application will automatically detect and load them
    
    ### Results:
    - Generated GIFs are saved to `./results/` directory
    - You can download each generated animation
    """)

# Footer
st.divider()
st.caption("MNIST Diffusion Generator | Built with Streamlit and PyTorch")