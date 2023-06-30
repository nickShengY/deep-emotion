# ... other imports ...
from data_preprocessing import preprocess_data, visualize_data
from model_training import create_model, train_model
from evaluate_test import evaluate_model, predict_class, save_model, load_model

def main():
    data_dir = 'data'
    model_path = './models/imageclassifier.h5'
    test_image_path = 'happygroup.jpg'

    # preprocess and visualize data
    train, val, test = preprocess_data(data_dir)
    visualize_data(data_dir)

    # create and train model
    model = create_model()
    hist = train_model(model, train, val)

    # evaluate model
    evaluate_model(model, test)

    # test on a single image
    predict_class(model, test_image_path)

    # save and load model
    save_model(model, model_path)
    new_model = load_model(model_path)
    predict_class(new_model, test_image_path)

if __name__ == "__main__":
    main()
