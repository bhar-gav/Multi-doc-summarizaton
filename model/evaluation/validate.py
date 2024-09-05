from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Example true and predicted labels for a classification task
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # True labels
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Predicted labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
