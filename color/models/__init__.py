import color.models.predict_color as predict_color_models
import color.models.predict_name as predict_name_models

_model_dict = {
    'predict_color_lstm': (predict_color_models.ColorPredictorLSTM, predict_color_models.ColorPredictionTraining),
    'predict_color_rnn': (predict_color_models.ColorPredictorRNN, predict_color_models.ColorPredictionTraining),
    'predict_color_cnn': (predict_color_models.ColorPredictorCNN, predict_color_models.ColorPredictionTraining),
    'predict_name_lstm': (predict_name_models.NamePredictorLSTM, predict_name_models.NamePredictionTraining),
    'predict_name_rnn': (predict_name_models.NamePredictorRNN, predict_name_models.NamePredictionTraining)
}


def get_model(key):
    return _model_dict[key]
