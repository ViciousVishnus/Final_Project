from mynn.layers.dense import dense
from mynn.initializers.normal import normal
from mynn.optimizers.sgd import SGD
from mynn.losses.cross_entropy import softmax_cross_entropy
from mynn.activations.relu import relu
from mynn.initializers.he_normal import he_normal
import mygrad as mg
import numpy as np
from mygrad.nnet.activations import sigmoid
final_counter = {"vishnu":0,"michael":0, "eden":0}
def euclidean_distance(arr1,arr2):
    """
    Returns the euclidean distance between two matrices.
    """
    distance = np.sqrt(np.sum((arr1 - arr2)**2))
    return distance
#np.save("Uknown_Walker",bm.get_test_data())
unknown_data = np.load("Unknown_Walker_e.npy")
def l1_loss(preds,ans):
    l_val = mg.sum(mg.abs(preds-ans))
   # print(l_val)
    row,col = preds.shape
    return l_val/row

from mynn.optimizers.adam import Adam
import liveplot
for BodyPartIndex in range(9):
    for CoordIndex in range(1,2):
        m = np.mean(unknown_data[BodyPartIndex][CoordIndex])#normalizes
        s = np.std(unknown_data[BodyPartIndex][CoordIndex])
        unknown_data[BodyPartIndex][CoordIndex] -= m
        unknown_data[BodyPartIndex][CoordIndex] /= s
        print(unknown_data[BodyPartIndex][CoordIndex])
        class Model:
            def __init__(self):
                self.dense1 = dense(1, 20, weight_initializer=normal)
                self.dense2 = dense(20, 30, weight_initializer=normal)
                self.dense3 = dense(30, 20, weight_initializer=normal)
                self.dense4 = dense(20, 1, weight_initializer=normal)
            def __call__(self, x):
                ''' Forward data through the network.


                This allows us to conveniently initialize a model `m` and then send data through it
                to be classified by calling `m(x)`.

                Parameters
                ----------
                x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D)
                    The data to forward through the network.

                Returns
                -------
                mygrad.Tensor, shape=(N, 1)
                    The model outputs.
                '''

                #We pass our data through a dense layer, use the activation function relu and then pass it through our second dense layer
                return self.dense4(sigmoid(self.dense3(sigmoid(self.dense2(sigmoid(self.dense1(x)))))))

            def weights1(self):
                weights=self.dense1.weight.data
                return weights
            def weights2(self):
                weights=self.dense2.weight.data
                return weights
            def weights3(self):
                weights=self.dense3.weight.data
                return weights
            def weights4(self):
                weights=self.dense4.weight.data
                return weights
            @property
            def parameters(self):
                ''' A convenience function for getting all the parameters of our model. '''
                return self.dense1.parameters + self.dense2.parameters + self.dense3.parameters + self.dense4.parameters
        model = Model()
        optim = Adam(model.parameters)
        # Create the shape-(1000,1) training data
        
        train_data = mg.linspace(0,19,100).reshape(100,1)
        # Create your parameters using your function `create_parameters`; 
        # start off with N=10 for the number of model parameters
        # to use for defining F(x)
        

        # Set `batch_size = 25`: the number of predictions that we will make in each training step

        # Define the function `true_f`, which should just accept `x` and return `np.cos(x)`
        def true_f(x):
            return np.interp(x,np.linspace(0,19,20),unknown_data[BodyPartIndex][CoordIndex])
        batch_size = 10
        for epoch_cnt in range(200000):
            idxs = np.arange(len(train_data))  # -> array([0, 1, ..., 9999])
            # shuffles these indices; we will use this to draw random batches
            # from our training data
            np.random.shuffle(idxs)  

            for batch_cnt in range(0, len(train_data)//batch_size):
                batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
                batch = train_data[batch_indices]  # random batch of our training data

                # compute the predictions for this batch: F(x; w, b, v)
                #print(batch)
                preds = model(batch)
               # print(len(preds))
                # compute the true (a.k.a desired) values for this batch: f(x) 

                true = true_f(batch)
               # print(true)
                # compute the loss associated with our predictions
                loss = l1_loss(preds,true)

                # back-propagate through your computational graph through your loss
                # this will compute: dL/dw, dL/db, dL/dv
                loss.backward()
                # execute your gradient descent function, passing all of your model's
                # parameters (w, b, v), and the learning rate. This will update your
                # model's parameters based on the loss that was computed
                optim.step()
                weights1 = model.weights1()
                weights2 = model.weights2()
                weights3 = model.weights3()
                weights4 = model.weights4()
                # FOR THE LOVE OF ALL THAT IS GOOD: NULL YOUR GRADIENTS
                loss.null_gradients()
                # this will plot add your loss value to the liveplot. Note that
                # `loss.item()` returns a float - we can't pass the plotter a 0D-Tensor
        #Loads person's weights
        for person in list(final_counter.keys()):
            c = "x"
            if(CoordIndex==1):
                c="y"
            data = np.load("walker_data/"+person+"/"+person+"_"+str(BodyPartIndex)+"_"+c+".npy")
            #adds distance between weights and data to counter
            #higher weights result in higher scores
            final_counter[person]+=euclidean_distance(weights1,data[0])
            final_counter[person]+=euclidean_distance(weights2,data[1])
            final_counter[person]+=euclidean_distance(weights3,data[2])
            final_counter[person]+=euclidean_distance(weights4,data[3])
mindex = np.argsort(np.array(list(final_counter.values())))[0]
string = list(final_counter.keys())[mindex]#takes the lowers value (the one closest to the unknown person)
print(final_counter)
print("GAIT Identified: "+string[0].upper()+string[1:])