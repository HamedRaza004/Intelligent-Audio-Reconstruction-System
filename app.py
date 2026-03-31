import streamlit as st
import tempfile, time, os

st.set_page_config(page_title="AI Audio Enhancement", layout="wide")
st.title("Intelligent Audio Reconstruction System")
st.caption("Powered by Facebook DNS64 · Deep Neural Speech Enhancement")

uploaded = st.file_uploader("Upload a noisy audio file (.wav)", type=["wav"])

if uploaded is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.audio(uploaded)
    
    if st.button("Enhance Audio", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fin:
            fin.write(uploaded.read())
            in_path = fin.name
        
        out_path = in_path.replace(".wav", "_enhanced.wav")
        
        with st.spinner("Running DNS64 neural model..."):
            try:
                t0 = time.time()
                from model import enhance_audio
                enhance_audio(in_path, out_path)
                elapsed = round(time.time() - t0, 2)
                st.success(f"Enhancement complete in {elapsed}s")
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()
        
        with col2:
            st.subheader("Enhanced")
            st.audio(out_path)
        
        try:
            from metrics import compute_metrics
            m = compute_metrics(in_path, out_path)
            st.subheader("Enhancement Metrics")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("SNR Improvement", f"{m['snr_improvement_db']} dB")
            mc2.metric("Noise Reduction", f"{m['noise_reduction_db']} dB")
            mc3.metric("Duration", f"{m['duration_sec']} s")
        except Exception as e:
            st.warning(f"Metrics unavailable: {e}")
        
        try:
            from visualize import make_waveform_plot, make_spectrogram_plot
            st.subheader("Waveform Comparison")
            st.image(make_waveform_plot(in_path, out_path), use_column_width=True)
            st.subheader("Spectrogram Comparison")
            st.image(make_spectrogram_plot(in_path, out_path), use_column_width=True)
        except Exception as e:
            st.warning(f"Visualizations unavailable: {e}")
        
        with open(out_path, "rb") as f:
            st.download_button(
                "Download Enhanced Audio",
                f,
                file_name="enhanced_output.wav",
                mime="audio/wav"
            )
        
        os.unlink(in_path)
        