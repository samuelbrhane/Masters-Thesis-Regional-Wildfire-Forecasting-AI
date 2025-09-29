from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Add, Concatenate,
    Embedding, MultiHeadAttention, Lambda
)
from constants import MONTHS_IN_DATA, DAYS_IN_WEEK

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def build_transformer_model(params, input_shapes):
    numeric_input = Input(shape=input_shapes['numeric'], name='numeric_input')
    month_input = Input(shape=input_shapes['month'], dtype='int32', name='month_input')
    dow_input = Input(shape=input_shapes['dow'], dtype='int32', name='dow_input')

    month_embed = Embedding(input_dim=MONTHS_IN_DATA, output_dim=params['month_embed_dim'])(month_input)
    dow_embed = Embedding(input_dim=DAYS_IN_WEEK, output_dim=params['dow_embed_dim'])(dow_input)

    x = Concatenate()([numeric_input, month_embed, dow_embed])
    x = LayerNormalization(epsilon=1e-6)(x)

    for _ in range(params['num_layers']):
        x = transformer_encoder(
            x,
            head_size=params['d_model'],
            num_heads=params['num_heads'],
            ff_dim=params['ff_dim'],
            dropout=params['dropout_rate']
        )

    x = Lambda(lambda x: x[:, -1, :])(x)
    x = Dropout(params['dropout_rate'])(x)

    output = Dense(1)(x)

    return Model(inputs=[numeric_input, month_input, dow_input], outputs=output)
