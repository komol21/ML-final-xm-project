import gradio as gr
import pandas as pd
import joblib

model = joblib.load('best_insurance_model.pkl')

def predict_insurance_charges(age, sex, bmi, children, smoker, region):
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'bmi': [bmi],
        'children': [children], 'smoker': [smoker], 'region': [region]
    })
    prediction = model.predict(input_data)[0]
    return f"**Predicted Insurance Charges:** ${prediction:,.2f}"

demo = gr.Interface(
    fn=predict_insurance_charges,
    inputs=[
        gr.Slider(minimum=18, maximum=100, value=30, step=1, label="Age"),
        gr.Radio(choices=["male", "female"], value="male", label="Sex"),
        gr.Slider(minimum=10, maximum=60, value=25, step=0.1, label="BMI"),
        gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Children"),
        gr.Radio(choices=["yes", "no"], value="no", label="Smoker"),
        gr.Dropdown(choices=["northeast", "northwest", "southeast", "southwest"], value="northeast", label="Region")
    ],
    outputs=gr.Markdown(),
    title="Insurance Charges Predictor",
    examples=[[25, "male", 22.5, 0, "no", "northeast"], [45, "female", 28.0, 2, "no", "southwest"]]
)

if __name__ == "__main__":
    demo.launch()
