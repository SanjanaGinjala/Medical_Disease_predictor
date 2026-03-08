from predict import predict_heart_disease

# sample patient data
sample_data = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]

result = predict_heart_disease(sample_data)

print("Prediction Result:", result)