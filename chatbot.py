# Chatbot logic

# Third party libraries
import numpy as np
from keras.models import Model

# Local libraries
from constants import MODEL_PATH


def get_model() -> Model:
    """ Loads and setup our pretrained model"""
    training_model = load_model(MODEL_PATH)

    encoder_inputs = training_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    latent_dim = 256
    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)

    return Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def decode_response(test_input: np.array) -> str:
    """ Predict a response based on the input"""
    # We get our pretrained model
    decoder_model = get_model()

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


def do_predict(input: np.array) -> str:
    return decode_response(input)
