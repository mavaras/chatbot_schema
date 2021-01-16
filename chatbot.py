# Chatbot logic

# Standard libraries
import re
from typing import (
    Dict,
    List,
    Tuple
)

# Third party libraries
import numpy as np
from tensorflow import keras
from keras.layers import (
    Dense,
    Input,
    LSTM as lstm
)
from keras.models import (
    Model
)

# Local libraries
from constants import (
    ANSWERS_PATH,
    QUESTONS_PATH
)


def get_data(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
    data = [re.sub(r"\[\w+\]", 'hi' if 'questions' in path else '', line) for line in data]
    data = [' '.join(re.findall(r"\w+",line)) for line in data]
    data = [line.lower() for line in data]

    return data


def get_questions_and_answers() -> List[Tuple[str]]:
    questions = get_data(QUESTONS_PATH)
    answers = get_data(ANSWERS_PATH)
    return list(zip(questions, answers))


def get_tokens(data: List[Tuple[str]]) -> Tuple[List[str]]:
    input_docs = []
    input_tokens = set()
    target_tokens = set()
    for line in data:
        input_doc, target_doc = line[0], line[1]
        input_docs.append(input_doc)

        target_doc = ' '.join(re.findall(r"[\w']+|[^\s\w]", target_doc))
        target_doc = f'<INICIO> {target_doc} <FINAL>'

        for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
            if token not in input_tokens:
                input_tokens.add(token)
        for token in target_doc.split():
            if token not in target_tokens:
                target_tokens.add(token)
    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))

    return input_tokens, target_tokens


def get_features_dicts(input_tokens, target_tokens) -> Tuple[Dict[str, str]]:
    input_features_dict = dict(
        [(token, i) for i, token in enumerate(input_tokens)]
    )
    target_features_dict = dict(
        [(token, i) for i, token in enumerate(target_tokens)]
    )

    return input_features_dict, target_features_dict


def get_model(trained_model: Model, num_decoder_tokens: int) -> Model:
    """ Loads and setup our pretrained model"""
    dim = 256

    decoder_lstm = lstm(dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')

    encoder_inputs = trained_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = trained_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_state_input_hidden = Input(shape=(dim,))
    decoder_state_input_cell = Input(shape=(dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(
        decoder_inputs,
        initial_state=decoder_states_inputs
    )
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


def string_to_matrix(text: str, input_features_dict: Dict[str, str], num_encoder_tokens: int) -> np.array:
    tokens = re.findall(r"[\w']+|[^\s\w]", text)
    max_encoder_seq_length = 26  # words limit per q/a
    user_input_matrix = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32'
    )
    for timestep, token in enumerate(tokens):
        if token in input_features_dict:
            user_input_matrix[0, timestep, input_features_dict[token]] = 1.

    return user_input_matrix


def decode_response(model: Model, test_input: np.array) -> str:
    """ Predict a response based on the input"""
    # Needed variables
    input_tokens, target_tokens = get_tokens(get_questions_and_answers())
    input_features_dict, target_features_dict = get_features_dicts(input_tokens, target_tokens)
    reverse_target_features_dict = dict(
        (i, token) for token, i in target_features_dict.items()
    )

    num_decoder_tokens = len(list(target_features_dict.items()))
    max_decoder_seq_length = 48
    test_input = string_to_matrix(test_input, input_features_dict, len(input_tokens))

    # We get our encoder & decoder models based on our pretrained model
    encoder_model, decoder_model = get_model(model, len(target_tokens))

    # Getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)

    # Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Setting the first token of target sequence with the start token
    target_seq[0, 0, target_features_dict['<INICIO>']] = 1.

    decoded_sentence = ''
    
    stop_condition = False
    while not stop_condition:
        # Predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)

        # Choosing the one with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += f' {sampled_token}'

        # Stop if hit max length or found the stop token
        stop_condition = sampled_token == '<FINAL>' or len(decoded_sentence) > max_decoder_seq_length

        # Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [hidden_state, cell_state]

    return decoded_sentence


def do_predict(model: Model, test_input: np.array) -> str:
    return decode_response(model, test_input)
