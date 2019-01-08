import os
from PIL import Image
import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_seq_length = 80

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

class CocoDataset_eval(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = os.path.join(self.root, path)
#         if self.transform is not None:
#             image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)    

def collate_eval_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

#     images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_eval_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset_eval(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_eval_fn)
    return data_loader
    

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=80):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        phrase_embed = []
        inputs = features.unsqueeze(1)
        inputs = inputs.to(device)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            phrase_embed.append(inputs.detach().cpu())
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
#         phrase_embed = torch.stack(phrase_embed)  
        sampled_ids = torch.stack(sampled_ids, 1)               # sampled_ids: (batch_size, max_seq_length)
        phrase_embed = torch.cat(phrase_embed,1).numpy()
        return sampled_ids,phrase_embed

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def captionidx_to_sentence(c,idx2word):
    sampled_caption = []
    for word_id in c:
        word = idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    return sentence

def pad_caption(c, max_length = 80):
        a = [int(x.unsqueeze(0)[0]) for x in list(c)]
        a += [int(x.unsqueeze(0)[0]) for x in list(torch.zeros((max_length-len(a)),dtype=torch.int64))]
    #     print(a)
        return a

class ImageCaptioning:
    def __init__(self,image_dir = './data/train2014/', new_image_dir = './data/resized2014/',
                vocab_path = 'data/vocab.pkl',caption_path = 'data/annotations/captions_train2014.json',
                model_path = 'models/', encoder_path = 'models/encoder-2-1000.ckpt',
                decoder_path = 'models/decoder-2-1000.ckpt',
                embed_size = 256, hidden_size = 512, num_layers = 1, max_seq_length = 80, set_up_data = False):
        
        (self.image_dir,self.new_image_dir,self.vocab_path,
        self.caption_path,self.model_path,self.encoder_path,
        self.decoder_path,self.max_seq_length) = (image_dir,new_image_dir,vocab_path,caption_path,
                                        model_path,encoder_path,decoder_path,max_seq_length)

        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

         # Build models
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, len(self.vocab), num_layers, max_seq_length).to(device)
        if encoder_path is not None: 
            self.image_dir = new_image_dir
            self.encoder = self.encoder.eval().to(device)
            self.encoder.load_state_dict(torch.load(encoder_path))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path))
        if not set_up_data:
            with open(self.vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
#             self.image_dir = self.new_image_dir    

    def set_up_data(self,image_size,threshold):

        self.image_size = image_size
        image_dir,new_image_dir,caption_path,vocab_path = (self.image_dir,self.new_image_dir,
                                                           self.caption_path,self.vocab_path)
        vocab = build_vocab(json=caption_path, threshold=threshold)
        self.vocab = vocab
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print("Total vocabulary size: {}".format(len(vocab)))
        print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
        resize_images(image_dir, new_image_dir, image_size)
        self.image_dir = self.new_image_dir

    def train(self,num_epochs=5,batch_size=128,crop_size=224,learning_rate=0.001,embed_size=256,
              hidden_size=512,num_layers=1,log_step=10,save_step=1000,num_workers=2):

        self.embed_size,self.hidden_size,self.num_layers = embed_size,hidden_size,num_layers
        image_dir,caption_path = self.image_dir,self.caption_path

        # Create model directory
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Image preprocessing, normalization for the pretrained resnet
        transform = transforms.Compose([ 
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])

        vocab = self.vocab

        # Build data loader
        data_loader = get_loader(image_dir,  caption_path, vocab, 
                                 transform,  batch_size,
                                 shuffle=True, num_workers= num_workers) 

        # Build the models
        encoder = EncoderCNN(embed_size).to(device)
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
        optimizer = torch.optim.Adam(params, lr= learning_rate)

        # Train the models
        total_step = len(data_loader)
        for epoch in range(num_epochs):
            for i, (images, captions, lengths) in enumerate(data_loader):

                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # Forward, backward and optimize
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)
                decoder.zero_grad()
                encoder.zero_grad()
                loss.backward()
                optimizer.step()

                # Print log info
                if i %  log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                          .format(epoch,  num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 

                # Save the model checkpoints
                if (i+1) %  save_step == 0:
                    self.encoder_path = os.path.join(self.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1))
                    self.decoder_path = os.path.join(self.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1))
                    torch.save(decoder.state_dict(),self.encoder_path)
                    torch.save(encoder.state_dict(),self.decoder_path)    

    def get_caption(self,image,show=True):

        encoder,decoder = self.encoder,self.decoder
        # Image preprocessing
        transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])

        vocab = self.vocab
        image_ = image.resize([224, 224], Image.LANCZOS)
        image_tensor = transform(image_).unsqueeze(0).to(device)

        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids,caption_embedding = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sentence = captionidx_to_sentence(sampled_ids,vocab.idx2word)
        sentence = sentence.replace('<start>', '').replace('<end>', '')

        # Print out the image and the generated caption
        print()
        print (sentence)
        if show:
            plt.imshow(np.asarray(image_))
            plt.show()
        self.decoder_embedding = decoder.embed
        return sentence,caption_embedding,image_

    def set_up_embeddings(self,data_size=50000,captions_path = "similar/50000_captions.pkl",
        image_paths = "similar/50000_images.pkl",knn_path = 'similar/knn_captions_model.joblib'):
        
        if knn_path is not None:
            self.nbrs = load(knn_path)
            with open(captions_path, "rb") as fp:
                self.captions = pickle.load(fp)
            with open(image_paths, "rb") as fp:
                self.image_paths = pickle.load(fp)
            return        

        image_dir,caption_path,max_seq_length,decoder_embedding = (self.image_dir,self.caption_path,
                                                                   self.max_seq_length,self.decoder_embedding)
        vocab = self.vocab
        batch_size = 1
        data_loader = get_eval_loader(image_dir, caption_path, vocab, 
                                None,  batch_size,
                                 shuffle=True, num_workers= 0) 

        captions = []
        captions_embeddings = []
        images = []
        for i in range(data_size):
        # for image,caption,l in data_loader:
            if i % 5000 == 0:
                print('i = {}, remaining = {}'.format(i,(data_size - i)))
            image,caption,l = next(iter(data_loader))
            caption = pad_caption(caption[0], max_seq_length)
            captions.append(caption)
            captions_embeddings.append(torch.cat(
                [decoder_embedding(torch.tensor(x).unsqueeze(0)).detach() for x in caption],1).\
                                       squeeze(0).numpy())
    #         t = transforms.ToPILImage()
    #         img = t(image.squeeze(0))
            images.append(image[0])
        #     plt.imshow(img)
        #     plt.show()
        
        
        with open("{}_embeddings.txt".format(data_size), "wb") as fp:   #Pickling
            pickle.dump(captions_embeddings, fp)
        with open("{}_captions.txt".format(data_size), "wb") as fp:   #Pickling
            pickle.dump(captions, fp)
        with open("{}_images.txt".format(data_size), "wb") as fp:   #Pickling
            pickle.dump(images, fp)
            
        self.captions_embeddings,self.captions,self.image_paths = captions_embeddings, captions, images
        
        self.nbrs = NearestNeighbors(n_neighbors = n).fit(self.captions_embeddings)    

        return #captions_embeddings,captions,images
    
    def save_KNN(self,n):
        dump(self.nbrs, '{}_knn_captions_model.joblib'.format(len(self.captions_embeddings))) 

    def get_similar_images(self,image):
        
        #captions_embeddings,captions,images,nbrs = (self.captions_embeddings,self.captions,
         #                                           self.image_paths,self.nbrs)
            
        _,caption_embedding,_ = self.get_caption(image,show=False)
        
        captions,images,nbrs = (self.captions,self.image_paths,self.nbrs)
        vocab = self.vocab
        distances, indices = nbrs.kneighbors(caption_embedding)
        indices = indices[0]
        imgs = []
        captions_filtered = []
        fig, axs = plt.subplots(nrows=nbrs.n_neighbors, sharex=True, figsize=(5, 20))
        for i in range(nbrs.n_neighbors):
            img = images[indices[i]]
            caption = captionidx_to_sentence(captions[indices[i]],vocab.idx2word)
            caption = caption.replace('<start>', '').replace('<end>', '')
            imgs.append(img)
            captions_filtered.append(caption)
            axs[i].set_title(caption)
            axs[i].imshow(Image.open(img))
        return imgs,captions_filtered,fig