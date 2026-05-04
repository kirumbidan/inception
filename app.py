import gradio as gr
import joblib

model = joblib.load("model.pkl")

def classify_ticket(category, instruction):
    #Pipeline handles Preprocessing and Transformation automatically
    input_df = pd.DataFrame([[instruction, category]], columns=['instruction', 'category'])
    prediction_probs = model.predict_proba(input_df)[0]

    class_names = model.classes_
    confidences = {class_names[i]: float(prediction_probs[i]) for i in range(len(class_names))}

    return confidences

ui = gr.Interface(
    fn=classify_ticket,
    inputs=[
        gr.Dropdown(["ORDER", "SHIPPING", "CANCEL", "INVOICE", "PAYMENT",
                     "REFUND", "FEEDBACK", "CONTACT", "ACCOUNT", "DELIVERY", "SUBSCRIPTION"], label="category"),
        gr.Textbox(lines=5, placeholder="Enter the user's intent here...", label="instruction")
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Support Ticket Classifier",
    description="Select a category and paste the ticket text to see the predicted classification."
)

ui.launch()