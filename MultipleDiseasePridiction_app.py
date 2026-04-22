from pathlib import Path
import pickle

import streamlit as st
from streamlit_option_menu import option_menu

from project_config import APP_SECTIONS, BASE_DIR, HOME_GIF, MODELS_DIR


st.set_page_config(
    page_title="MEDCHECK - Multiple Disease Prediction",
    layout="wide",
    page_icon="🧑‍⚕️",
)


st.markdown(
    """
    <style>
        [data-testid="stDecoration"] {
            background: linear-gradient(90deg, #0f4c81, #f6fbff);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eff7ff 0%, #f9fcff 100%);
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(15, 76, 129, 0.08), transparent 25%),
                linear-gradient(180deg, #f7fbff 0%, #edf5fb 100%);
        }
        .hero-card, .info-card {
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(15, 76, 129, 0.12);
            box-shadow: 0 18px 45px rgba(15, 76, 129, 0.08);
            margin-bottom: 1rem;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.4rem 0;
        }
        .metric-card {
            border-radius: 16px;
            padding: 0.9rem 1rem;
            background: linear-gradient(135deg, #0f4c81, #2a7ab9);
            color: white;
        }
        .metric-card h4, .metric-card p {
            margin: 0;
        }
        div.stButton > button:first-child,
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(90deg, #0f4c81, #1f6ca8);
            color: white;
            border-radius: 999px;
            border: none;
        }
        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            color: white;
            border: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    missing_models = []

    for section in APP_SECTIONS:
        model_path = MODELS_DIR / section["model_filename"]
        if not model_path.exists():
            missing_models.append(model_path.name)
            continue

        with model_path.open("rb") as model_file:
            models[section["menu_label"]] = pickle.load(model_file)

    if missing_models:
        raise FileNotFoundError(
            "Missing model files: "
            + ", ".join(missing_models)
            + ". Run `./mdp/bin/python train_models.py` first."
        )

    return models


def render_input(field, key: str):
    if field["type"] in {"binary", "select"}:
        selected = st.selectbox(
            field["label"],
            options=list(field["options"].keys()),
            index=field.get("default_index", 0),
            help=field.get("help"),
            key=key,
        )
        return [(field["name"], float(field["options"][selected]))]

    if field["type"] == "vector_select":
        selected = st.selectbox(
            field["label"],
            options=list(field["options"].keys()),
            index=field.get("default_index", 0),
            help=field.get("help"),
            key=key,
        )
        return [
            (feature_name, float(feature_value))
            for feature_name, feature_value in zip(field["feature_names"], field["options"][selected])
        ]

    return [
        (
            field["name"],
            float(
                st.number_input(
            field["label"],
            min_value=field.get("min_value", 0.0),
            max_value=field.get("max_value", 1000.0),
            value=field.get("default", 0.0),
            step=field.get("step", 1.0),
            help=field.get("help"),
            key=key,
                )
            ),
        )
    ]


def render_prediction_page(section, model):
    st.markdown(
        f"<div class='hero-card'><h1 style='color:#0f4c81; margin-bottom:0.3rem;'>{section['title']}</h1>"
        f"<p style='margin:0; color:#35546f;'>{section['description']}</p></div>",
        unsafe_allow_html=True,
    )

    st.caption(section["input_hint"])

    with st.form(section["form_key"]):
        values = []
        submitted_values = {}
        columns = st.columns(section["column_count"])

        for index, field in enumerate(section["fields"]):
            with columns[index % section["column_count"]]:
                field_values = render_input(field, f"{section['form_key']}_{field['name']}")
                for feature_name, value in field_values:
                    values.append(value)
                    submitted_values[feature_name] = value

        submitted = st.form_submit_button("Predict")

    if not submitted:
        st.info(section["idle_message"])
        return

    prediction = int(model.predict([values])[0])
    diagnosis = (
        section["messages"]["positive"]
        if prediction == section.get("positive_class", 1)
        else section["messages"]["negative"]
    )

    if prediction == section.get("positive_class", 1):
        st.error(diagnosis)
    else:
        st.success(diagnosis)

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba([values])[0][prediction])
        st.caption(f"Model confidence for this prediction: {confidence:.2%}")

    st.write("Submitted values")
    st.json(submitted_values)


def render_home():
    st.markdown(
        """
        <div class='hero-card'>
            <h1 style='color:#0f4c81; margin-bottom:0.3rem;'>MEDCHECK</h1>
            <p style='font-size:1.05rem; color:#35546f; margin-bottom:0.6rem;'>
                A Streamlit-based multiple disease prediction system for diabetes, heart disease,
                Parkinson's disease, and kidney disease.
            </p>
            <p style='color:#35546f; margin:0;'>
                This interface now uses guided inputs so users can enter values safely without
                remembering every numeric encoding from the notebooks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='metric-strip'>
            <div class='metric-card'><h4>4 models</h4><p>One workflow for four prediction screens</p></div>
            <div class='metric-card'><h4>Local datasets</h4><p>Training and inference stay inside this project</p></div>
            <div class='metric-card'><h4>Shared config</h4><p>App inputs and model paths are defined in one place</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if HOME_GIF.exists():
        st.image(str(HOME_GIF), use_container_width=True)

    st.markdown(
        """
        <div class='info-card'>
            <strong>How to use the app</strong>
            <p style='margin:0.4rem 0 0 0; color:#35546f;'>
                Pick a disease from the left sidebar, fill in the clinical inputs, and press
                <code>Predict</code>. Use the project README for setup, retraining, and notebook notes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    try:
        models = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        selected = option_menu(
            "MEDCHECK",
            ["Home Page"] + [section["menu_label"] for section in APP_SECTIONS],
            menu_icon="hospital-fill",
            icons=["house", "activity", "heart", "person", "droplet"],
            default_index=0,
            styles={"nav-link-selected": {"background-color": "#0f4c81"}},
        )

        st.caption(f"Project root: {BASE_DIR.name}")

    if selected == "Home Page":
        render_home()
        return

    for section in APP_SECTIONS:
        if selected == section["menu_label"]:
            render_prediction_page(section, models[selected])
            break


if __name__ == "__main__":
    main()
