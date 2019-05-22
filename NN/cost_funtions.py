def square_error(tagged_dataset, evaluation_funtion):
    data_length = len(tagged_dataset)
    error = 0
    for sample, dataset_tag in tagged_dataset:
        predicted_tag = evaluation_funtion(sample)
        error += (dataset_tag - predicted_tag)**2
    return error / data_length