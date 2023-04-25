from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset



class Dataset_with_annotations(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode='train', transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.wikiart_dataset = load_dataset("huggan/wikiart")
        self.wikiart_dataset.set_format(type="torch", columns=['image', 'artist', 'genre', 'style'])
        print(self.wikiart_dataset)
        if mode == 'train':
            self.wikiart_dataset = self.wikiart_dataset['train'][:64000]
        else: 
            self.wikiart_dataset = self.wikiart_dataset['train'][64000:]
       
        #Dictionary = {"dataset":"huggan/wikiart","config":"huggan--wikiart","split":"train","features":[{"feature_idx":0,"name":"image","type":{"_type":"Image"}},{"feature_idx":1,"name":"artist","type":{"names":["Unknown Artist","boris-kustodiev","camille-pissarro","childe-hassam","claude-monet","edgar-degas","eugene-boudin","gustave-dore","ilya-repin","ivan-aivazovsky","ivan-shishkin","john-singer-sargent","marc-chagall","martiros-saryan","nicholas-roerich","pablo-picasso","paul-cezanne","pierre-auguste-renoir","pyotr-konchalovsky","raphael-kirchner","rembrandt","salvador-dali","vincent-van-gogh","hieronymus-bosch","leonardo-da-vinci","albrecht-durer","edouard-cortes","sam-francis","juan-gris","lucas-cranach-the-elder","paul-gauguin","konstantin-makovsky","egon-schiele","thomas-eakins","gustave-moreau","francisco-goya","edvard-munch","henri-matisse","fra-angelico","maxime-maufra","jan-matejko","mstislav-dobuzhinsky","alfred-sisley","mary-cassatt","gustave-loiseau","fernando-botero","zinaida-serebriakova","georges-seurat","isaac-levitan","joaquã­n-sorolla","jacek-malczewski","berthe-morisot","andy-warhol","arkhip-kuindzhi","niko-pirosmani","james-tissot","vasily-polenov","valentin-serov","pietro-perugino","pierre-bonnard","ferdinand-hodler","bartolome-esteban-murillo","giovanni-boldini","henri-martin","gustav-klimt","vasily-perov","odilon-redon","tintoretto","gene-davis","raphael","john-henry-twachtman","henri-de-toulouse-lautrec","antoine-blanchard","david-burliuk","camille-corot","konstantin-korovin","ivan-bilibin","titian","maurice-prendergast","edouard-manet","peter-paul-rubens","aubrey-beardsley","paolo-veronese","joshua-reynolds","kuzma-petrov-vodkin","gustave-caillebotte","lucian-freud","michelangelo","dante-gabriel-rossetti","felix-vallotton","nikolay-bogdanov-belsky","georges-braque","vasily-surikov","fernand-leger","konstantin-somov","katsushika-hokusai","sir-lawrence-alma-tadema","vasily-vereshchagin","ernst-ludwig-kirchner","mikhail-vrubel","orest-kiprensky","william-merritt-chase","aleksey-savrasov","hans-memling","amedeo-modigliani","ivan-kramskoy","utagawa-kuniyoshi","gustave-courbet","william-turner","theo-van-rysselberghe","joseph-wright","edward-burne-jones","koloman-moser","viktor-vasnetsov","anthony-van-dyck","raoul-dufy","frans-hals","hans-holbein-the-younger","ilya-mashkov","henri-fantin-latour","m.c.-escher","el-greco","mikalojus-ciurlionis","james-mcneill-whistler","karl-bryullov","jacob-jordaens","thomas-gainsborough","eugene-delacroix","canaletto"],"_type":"ClassLabel"}},{"feature_idx":2,"name":"genre","type":{"names":["abstract_painting","cityscape","genre_painting","illustration","landscape","nude_painting","portrait","religious_painting","sketch_and_study","still_life","Unknown Genre"],"_type":"ClassLabel"}},{"feature_idx":3,"name":"style","type":{"names":["Abstract_Expressionism","Action_painting","Analytical_Cubism","Art_Nouveau","Baroque","Color_Field_Painting","Contemporary_Realism","Cubism","Early_Renaissance","Expressionism","Fauvism","High_Renaissance","Impressionism","Mannerism_Late_Renaissance","Minimalism","Naive_Art_Primitivism","New_Realism","Northern_Renaissance","Pointillism","Pop_Art","Post_Impressionism","Realism","Rococo","Romanticism","Symbolism","Synthetic_Cubism","Ukiyo_e"],"_type":"ClassLabel"}}]}
        #artists = Dictionary['features'][1]['type']['names']
        #generes = Dictionary['features'][2]['type']['names']
        #styles = Dictionary['features'][3]['type']['names']
        #self.annotations = []
        #for i in range(len(self.wikiart_dataset['style'])):
        #  self.annotations.append('this image depicts a piece of art made by ' + str(artists[int(self.wikiart_dataset['artist'][i].data)]).replace("-", " " ).replace("_", " " ) + ', it belongs to the ' + str(generes[int(self.wikiart_dataset['genre'][i].data)]).replace("_", " " ).replace("-", " " )  + ' genre and has a ' + str(styles[int(self.wikiart_dataset['style'][i].data)]).replace("_", " " ).replace("-", " " )   + ' style')



    def __len__(self):
        return len(self.wikiart_dataset['style'])

    def __getitem__(self, idx):

      
        #return {'image': self.wikiart_dataset['image'][idx], 'caption': self.annotations[idx]}
        return {'images': self.wikiart_dataset['image'][idx], 'labels': self.wikiart_dataset['genre'][idx]}


if __name__ == "__main__":


    wikiart_dataset = Dataset_with_annotations()
    
  
    
    # dataset = dataset.map(...)  # do all your processing here

    #dataset.push_to_hub("Cmeo97/WikiArt_Annotated")

