import color.models.predict_color as predict_color_lstm

_model_dict = {
    'predict_color_lstm': (predict_color_lstm.ColorPredictorLSTM, predict_color_lstm.ColorPredictionTraining),
    'predict_color_rnn': (predict_color_lstm.ColorPredictorRNN, predict_color_lstm.ColorPredictionTraining),
    'predict_color_cnn': (predict_color_lstm.ColorPredictorCNN, predict_color_lstm.ColorPredictionTraining)
}


def get_model(key):
    return _model_dict[key]
