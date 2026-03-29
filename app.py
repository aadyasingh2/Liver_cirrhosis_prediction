import gradio as gr
import pickle
import numpy as np

# 1. Load your model
model = pickle.load(open('liver_model_final.sav', 'rb'))

def predict_liver_disease(age, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio, gender):
    gender_num = 1 if gender == "Male" else 0
    input_array = np.array([[age, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio, gender_num]])
    
    # Get the probabilities for each class
    # proba will look like [[prob_of_class_1, prob_of_class_2]]
    proba = model.predict_proba(input_array)[0]
    
    # Assuming class 1 is Liver Patient and class 2 is Healthy
    # We take the probability of the first class (index 0)
    risk_percent = proba[0] * 100 
    
    if risk_percent > 50:
        return f"⚠️ HIGH RISK: {risk_percent:.2f}% Probability"
    else:
        return f"✅ LOW RISK: {100 - risk_percent:.2f}% Healthy Probability"

# 2. Define your Input components separately so we can "Clear" them
inputs = [
    gr.Slider(0, 100, label="Age"),
    gr.Number(label="Total Bilirubin"),
    gr.Number(label="Direct Bilirubin"),
    gr.Number(label="Alkphos"),
    gr.Number(label="Sgpt (ALT)"),
    gr.Number(label="Sgot (AST)"),
    gr.Number(label="Total Proteins"),
    gr.Number(label="Albumin"),
    gr.Number(label="A/G Ratio"),
    gr.Radio(["Male", "Female"], label="Gender")
]
output = gr.Textbox(label="AI Diagnosis Result")

# 3. Create the Blocks Layout (This gives you more control than gr.Interface)
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Liver Disease Diagnostic System")
    gr.Markdown("Extra Trees Classifier | 74.22% Accuracy")
    
    with gr.Row():
        with gr.Column():
            # Add all inputs to the left column
            for i in inputs:
                i.render()
        with gr.Column():
            # Add output and buttons to the right column
            output.render()
            btn = gr.Button("Submit", variant="primary")
            # This is your Clear Button
            clear = gr.ClearButton(components=inputs + [output])
            
    # Connect the Submit button to the function
    btn.click(fn=predict_liver_disease, inputs=inputs, outputs=output)

demo.launch(share=True)