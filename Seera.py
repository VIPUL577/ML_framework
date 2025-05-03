from Seera_Engine import Tensor,np,autograd4nn
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle

class Layer:
    def __init__(self):
        # self.parameters ={}

        self.inputs=[]
        self.outputs=[]
        self.activations={"relu":Tensor.relu,
                          "sigmoid":Tensor.sigmoid,
                          "softmax":Tensor.softmax,
                          "tanh":Tensor.tanh
                          }
        
class Input(Layer):
    def __init__(self,shape):
        super().__init__()
        self.shape=shape
        X=Tensor.random(shape)
        self.weights=0
        self.bais=0
        self.inputs=X
    def forward(self):
        self.outputs=self.inputs
        return self.outputs     
    
    def __repr__(self):
        return f"Input Layer with {self.inputs.shape[0]} units"
        
class Dense(Layer):
    
    def __init__(self,
                 in_units,out_units,
                 activation=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 ):
        
        super().__init__()
        initializers = {
                        "zeros": 0,
                        "ones": 1,
                        "random_normal": 1,  # std = 1, standard normal
                        "random_uniform": 0.05,  # typical default range [-0.05, 0.05]
                        
                        "he_normal": np.sqrt(2 / in_units)*0.1,
                        "he_uniform": np.sqrt(6 / in_units)*0.1,
                        
                        "glorot_normal": np.sqrt(2 / (in_units + out_units))*0.1,
                        "glorot_uniform": np.sqrt(6 / (in_units + out_units))*0.1,

                        "lecun_normal": np.sqrt(1 / in_units)*0.1,
                        "lecun_uniform": np.sqrt(3 / in_units)*0.1
                                                }

        if activation not in self.activations.keys():
            raise ValueError('''The activation should be "relu", "sigmoid" or "softmax" ''')
        if kernel_initializer not in initializers.keys():
            raise ValueError(f'''The initializer should be {initializers.keys()} ''')
        self.layeract=activation
        self.in_units=in_units
        self.out_units=out_units
        self.weights=Tensor.random((in_units,out_units))*initializers[kernel_initializer]
        self.bais=Tensor.random((out_units,1))*initializers[bias_initializer]
        
    def __call__(self, input_layer):
        if not isinstance(input_layer,Layer):
            raise TypeError("The input should be a Layer")
        self.inpobj = input_layer
        return self

    def forward(self):
              
        self.inputs=self.inpobj.outputs
        self.z=self.weights.T().matmul(self.inputs)+self.bais
        self.outputs=self.activations[self.layeract](self.z)

        return self.outputs
        
    def update_params(self,vW,vB):
        self.weights.value=self.weights.value-(vW.T)
        self.bais.value=self.bais.value-(vB)

    def get_weights(self):
        return self.weights,self.bais
    
    def set_weights(self,W,B):
        print(W,self.weights,B,self.bais)
        if W.shape==self.weights.shape and B.shape==self.bais.shape :
            self.weights=W
            self.bais=B
        else:
            raise TypeError('''The parameters should be of appropriate shape and should be "Tensor"''')
        
    def __repr__(self):
        return f"Dense Layer with {self.weights.shape[1]*(self.weights.shape[0]+1)} parameters and activation {self.layeract}"
  
class Conv2D(Layer):
    def __init__(self,out_channels,in_channels,kernel_size,activation,stride=1,zero_padding=0,kernel_initializer="he_normal",bias_initializer="zeros"):
        super().__init__()
        initializers = {
                        "zeros": 0,
                        "ones": 1,
                        "random_normal": 1,  # std = 1, standard normal
                        "random_uniform": 0.05,  # typical default range [-0.05, 0.05]
                        
                        "he_normal": np.sqrt(2 / in_channels*kernel_size[0]*kernel_size[1])*0.1,
                        "he_uniform": np.sqrt(6 / in_channels*kernel_size[0]*kernel_size[1])*0.1,
                        
                        "glorot_normal": np.sqrt(2 / (in_channels + out_channels)*kernel_size[0]*kernel_size[1])*0.1,
                        "glorot_uniform": np.sqrt(6 / (in_channels + out_channels)*kernel_size[0]*kernel_size[1])*0.1,

                        "lecun_normal": np.sqrt(1 / in_channels*kernel_size[0]*kernel_size[1])*0.1,
                        "lecun_uniform": np.sqrt(3 / in_channels*kernel_size[0]*kernel_size[1])*0.1
                    }
        
        if activation not in self.activations.keys():
            raise ValueError('''The activation should be "relu", "sigmoid" or "softmax" ''')
        if kernel_initializer not in initializers.keys():
            raise ValueError(f'''The initializer should be {initializers.keys()} ''')
        if not isinstance(stride or zero_padding,int):
            raise ValueError("stride and zero_padding should be integers")
        self.layeract=activation
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.zero_padding=zero_padding
        self.bais=0
        self .weights=Tensor(np.random.normal(loc=0,scale=.5,size=(out_channels,in_channels,kernel_size[0],
                                   kernel_size[1])),is_leaf=True)*initializers[kernel_initializer]

        
    def __call__(self,input_layer):
        if not isinstance(input_layer,Layer):
            raise TypeError("The input should be a Layer")
        self.inpobj = input_layer

        return self
    
    def forward(self):
        self.inputs=self.inpobj.outputs
        output_height = int((self.inputs.shape[1] - self.kernel_size[0] + 2 * self.zero_padding) / self.stride) + 1
        output_width = int((self.inputs.shape[2] - self.kernel_size[1] + 2 * self.zero_padding) / self.stride) + 1
        if isinstance(self.bais, int) or self.bais.shape != (self.out_channels, output_height, output_width):
            self.bais = Tensor.zeros((self.out_channels, output_height, output_width))
        self.z=Tensor.conv2d(self.inputs,self.weights)+self.bais
        self.outputs=self.activations[self.layeract](self.z)
        
        return self.outputs
    
    def set_weights(self,W,B): 
        if W.shape==self.weights.shape  :
            self.weights=W
            self.bais=B
               
    def update_params(self,vW,vB):
        # a=self.weights.value

        self.weights.value=self.weights.value-vW
        self.bais.value=self.bais.value-vB
        # b=self.weights.value
        # print(a-b)
    def get_weights(self):
        return self.weights,self.bais
    
    def __repr__(self):
        return f"Convolutional Layer with {self.weights.shape[0]} filters of shape {self.weights.shape[2],self.weights.shape[3]} and activation {self.layeract}"

#########################################
class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.flatten(self.inputs)
        return self.outputs

    def __repr__(self):
        return "Flatten Layer"
    
class UpSampling2D_Nearest(Layer):
    def __init__(self, size=(2, 2)):
        super().__init__()
        self.size = size

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.UpSample2Dnearest(self.inputs, size=self.size)
        return self.outputs

    def __repr__(self):
        return f"UpSampling2D Layer with size {self.size}, mode nearest"
    
class MaxPool2D(Layer):
    def __init__(self, pool_size=(2, 2), stride=1, padding=0):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.maxpool2d(self.inputs, kernelsize=self.pool_size, stride=self.stride, padding=self.padding)
        return self.outputs

    def __repr__(self):
        return f"MaxPool2D Layer with pool size {self.pool_size}, stride: {self.stride}, padding: {self.padding}"
    
    
class Concatenate(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, *input_layers):
        if len(input_layers)!=2:
            raise ValueError("Only 2 layers should be given, more than that are not supported in this version.")
        for layer in input_layers:
            if not isinstance(layer, Layer):
                raise TypeError("All inputs must be Layer instances.")
        self.input_layers = input_layers
        return self

    def forward(self):
        self.inputs = [layer.outputs for layer in self.input_layers]
        self.outputs = Tensor.concatenate(self.inputs[0],self.inputs[1])
        return self.outputs

    def __repr__(self):
        return f"Concatenate Layer along axis {self.axis}"


class Sequential:
    def __init__(self, layers):
        if not isinstance(layers, list):
            raise TypeError("The layers should be given as input as a list[]")
        if not isinstance(layers[0], Input):
            raise TypeError("The first layer should be an Input layer")
        layerlis=[layers[0]]
        for i,layer in enumerate(layers):
            if i==0:
                continue
            layerlis.append(layer(layers[i-1]))
        self.model = layerlis

    def forward(self, X=None):
        if X is not None:
            if not isinstance(X, Tensor):
                raise TypeError("The input should be a Tensor.")
            self.model[0].inputs = X
        output = None
        for layer in self.model:
            output = layer.forward()
        return output

    def predict(self, X):
        return self.forward(X)

    def zero_grad(self):
        """Reset gradients to zero"""
        for layer in self.model:
            if hasattr(layer, "update_params"):
                layer.weights.node.cp = 0
                layer.bais.node.cp = 0
                
    def get_params(self):
        """
        Updates the parameters of all trainable layers in the model.
        Parameters:
            learning_rate (float): The learning rate or update value to be subtracted from each parameter.
        """
        # We assume that only layers with an update_params method need to update their parameters.
        for layer in self.model:
            # Skip the input layer which may not have trainable parameters.
            if hasattr(layer, "update_params"):
                print(layer.get_weights())
                
    def summary(self):
        """
        Prints a summary of the model including each layer's information.
        """
        print("Model Summary:")
        for idx, layer in enumerate(self.model):
            # This depends on each layer having a sensible __repr__ implementation.
            print(f"Layer {idx}: {layer}")  
            
            
    def fit(self,X_train,y,Optimizer,Loss,Epochs,Loss_interval=20):
        history=np.array([])
        if X_train.shape[0]!=y.shape[0]:
            raise ValueError("the number of outputs shall be equal to number of inputs")
        for epoch in range (0,Epochs):
            lossavg=0
            for i in range(0,X_train.shape[0]):
                X=Tensor(X_train[i],is_leaf=True)
                ypred=self.forward(X)
                out=Tensor(y[i])
                loss=Loss(ypred,out)
                lossavg+=np.sum(loss.value)
                # history=np.append(history,loss.value)
                
                if (i%Loss_interval==0 ) and i>0:
                    print(f"Epoch no.{epoch} - Item_no.{i} - Loss: {lossavg/Loss_interval}")
                    history=np.append(history,lossavg/Loss_interval)
                    lossavg=0
                self.zero_grad()
                autograd4nn(loss)
                Optimizer.step()
        return history
    def save(self, filepath):
        """
        Save the model architecture and weights to a file.
        """
        model_data = []
        for layer in self.model:
            layer_info = {
                'class': layer.__class__.__name__,
                'config': layer.__dict__,
            }
            # Save weights and biases if the layer has them
            if hasattr(layer, 'get_weights'):
                weights, bais = layer.get_weights()
                # Convert to numpy arrays for serialization
                layer_info['weights'] = weights.value if hasattr(weights, 'value') else weights
                layer_info['bais'] = bais.value if hasattr(bais, 'value') else bais
            model_data.append(layer_info)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file and return a Sequential instance.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        layers = []
        layer_objs = {}
        for i, info in enumerate(model_data):
            class_name = info['class']
            config = info['config']
            # Remove weights/biases from config to avoid double-setting
            # config.pop('weights', None)
            # config.pop('bais', None)
            # Reconstruct layer

            if class_name == 'Input':
                layer = Input(config['shape'])
            elif class_name == 'Dense':
                layer = Dense(config['in_units'], config['out_units'], activation=config['layeract'])
                # print(config)
                layer.set_weights((config['weights']), (config['bais']))
            elif class_name == 'Conv2D':
                layer = Conv2D(config['out_channels'], config['in_channels'], config['kernel_size'], activation=config['layeract'])
                # print(config)
                
                layer.set_weights((config['weights']), (config['bais']))
            elif class_name == 'Flatten':
                layer = Flatten()
            elif class_name == 'UpSampling2D_Nearest':
                layer = UpSampling2D_Nearest(size=config['size'])
            elif class_name == 'MaxPool2D':
                layer = MaxPool2D(pool_size=config['pool_size'], stride=config['stride'], padding=config['padding'])
            elif class_name == 'Concatenate':
                layer = Concatenate()
            else:
                raise ValueError(f"Unknown layer type: {class_name}")

            layers.append(layer)
            layer_objs[i] = layer
        # Reconnect layers (for non-input layers)
        for i in range(1, len(layers)):
            if hasattr(layers[i], 'inpobj'):
                layers[i].inpobj = layers[i-1]
        return Sequential(layers)              
                

class Loss:
    def mse(self, y_pred,y):
        """
        Mean Squared Error.
        Loss = mean((y_pred - y_true)^2)
        """
        return (y_pred - y) ** 2

    def mae(self, y_pred,y):
        """
        Mean Absolute Error.
        Loss = mean(|y_pred - y_true|)
        """
        return (y_pred - y).abs()

    def binary_cross_entropy(self, y_pred,y, epsilon=1e-15):
        """
        Binary Cross-Entropy for binary classification.
        Applies clipping to avoid log(0).
        """
        return -y*(y_pred.log())-((1-y)*((1-y_pred).log()))
        
    def categorical_cross_entropy(self, y_pred,y, epsilon=1e-15):
        """
        Categorical Cross-Entropy for multi-class classification.
        Assumes y_true one-hot encoded and predictions as probability distributions.
        """
        return (-y*((y_pred+epsilon).log())).sum()
    
class SGD:
    def __init__(self,model,lr,momentum=0):
        if not isinstance(model,Sequential):
            raise TypeError("Model should be sequential data type")
        self.model=model
        self.lr=lr
        self.momentum=momentum
        self.velocity={}
        for layer in self.model.model:
            self.velocity[layer]=[0,0]#[weights,bais]
            
    def step(self):
        for layer in self.model.model:
            if hasattr(layer, "update_params"):
                g0=layer.weights.node.cp
                g1=layer.bais.node.cp

                self.velocity[layer][0]=(self.momentum*self.velocity[layer][0])+(self.lr*g0)
                self.velocity[layer][1]=(self.momentum*self.velocity[layer][1])+(self.lr*g1)
                # print(self.velocity[layer][0])
                layer.update_params(self.velocity[layer][0],self.velocity[layer][1])
         
class Adam:
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        if not isinstance(model, Sequential):
            raise TypeError("Model should be sequential data type")
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 1  # time step
        
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        
        for layer in self.model.model:
            self.m[layer] = [0, 0]  # [m_weights, m_biases]
            self.v[layer] = [0, 0]  # [v_weights, v_biases]

    def step(self):
        for layer in self.model.model:
            if hasattr(layer, "update_params"):
                g0 = layer.weights.node.cp
                g1 = layer.bais.node.cp

                # Update biased first moment estimate
                self.m[layer][0] = self.beta1 * self.m[layer][0] + (1 - self.beta1) * g0
                self.m[layer][1] = self.beta1 * self.m[layer][1] + (1 - self.beta1) * g1

                # Update biased second raw moment estimate
                self.v[layer][0] = self.beta2 * self.v[layer][0] + (1 - self.beta2) * (g0 ** 2)
                self.v[layer][1] = self.beta2 * self.v[layer][1] + (1 - self.beta2) * (g1 ** 2)

                # Compute bias-corrected first moment estimate
                m_hat0 = self.m[layer][0] / (1 - self.beta1 ** self.t)
                m_hat1 = self.m[layer][1] / (1 - self.beta1 ** self.t)

                # Compute bias-corrected second raw moment estimate
                v_hat0 = self.v[layer][0] / (1 - self.beta2 ** self.t)
                v_hat1 = self.v[layer][1] / (1 - self.beta2 ** self.t)

                # Update parameters
                update_w = (self.lr * m_hat0) / ((v_hat0 ** 0.5) + self.eps)
                update_b = (self.lr * m_hat1) / ((v_hat1 ** 0.5) + self.eps)

                layer.update_params(update_w, update_b)
        
        self.t += 1
