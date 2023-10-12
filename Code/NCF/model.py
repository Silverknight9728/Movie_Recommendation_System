
from sklearn import metrics, preprocessing
from tensorflow.keras import models, layers, utils  #(2.6.0)

def create_model(dataset, embed, opti, loss):
    in_shape = (1,)
    out_shape = (embed,)
    product_input = layers.Input(name="xproducts_in", shape=in_shape)
    user_input = layers.Input(name="xusers_in", shape=in_shape)
    
    user_embed_layer = layers.Embedding(name="nn_xusers_emb", input_dim=dataset.shape[0], output_dim=embed)(user_input)
    user_embed_out = layers.Reshape(name='nn_xusers', target_shape=out_shape)(user_embed_layer)
    ## embeddings and reshape
    movie_embed_layer = layers.Embedding(name="nn_xproducts_emb", input_dim=dataset.shape[1], output_dim=embed)(product_input)
    movie_embed_out = layers.Reshape(name='nn_xproducts', target_shape=out_shape)(movie_embed_layer)
    ## concat and dense
    nn_xx = layers.Concatenate()([user_embed_out, movie_embed_out])
    nn_xx = layers.Dense(name="nn_xx", units=int(embed/2), activation='relu')(nn_xx)
    nn_xx = layers.Dropout(0.1,name="nn_drop")(nn_xx)

    # Merge A & B
    y_out = layers.Dense(name="y_out", units=1, activation='linear')(nn_xx)
    # Compile
    model = models.Model(inputs=[user_input,product_input], outputs=y_out, name="Neural_CollaborativeFiltering")
    model.compile(optimizer=opti, loss=loss, metrics=['mean_absolute_error'])
    
    return model