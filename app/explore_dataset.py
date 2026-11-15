import os

train_path = "dataset/asl_alphabet_train/asl_alphabet_train"
test_path = "dataset/asl_alphabet_test/asl_alphabet_test"

print("TRAIN DATASET")
print("----------------")

classes = sorted(os.listdir(train_path))

print("Number of classes:", len(classes))
print("Classes:", classes)

print("\nImages per class:")
for class_name in classes:
    class_path = os.path.join(train_path, class_name)
    if os.path.isdir(class_path):
        print(class_name, ":", len(os.listdir(class_path)))

print("\nTEST DATASET")
print("----------------")
print("Number of test images:", len(os.listdir(test_path)))