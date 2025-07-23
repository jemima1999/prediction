import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. Text Encoder simple ===
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super(TextEncoder, self).__init__()
        # Embedding : transforme chaque mot en vecteur
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM : réseau séquentiel pour comprendre la phrase
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, text_indices):
        # text_indices : [batch_size, seq_len]
        x = self.embedding(text_indices)  # [batch_size, seq_len, embed_dim]
        _, (hidden, _) = self.lstm(x)    # hidden : [1, batch_size, hidden_dim]
        sentence_feature = hidden.squeeze(0)  # [batch_size, hidden_dim]
        return sentence_feature

# === 2. Style Encoder ===
class StyleEncoder(nn.Module):
    def __init__(self, style_vocab_size, style_embed_dim=64, output_dim=256):
        super(StyleEncoder, self).__init__()
        # Embedding pour style (ex : "wax", "mariage", "afro-chic")
        self.embedding = nn.Embedding(style_vocab_size, style_embed_dim)
        # FC : couche fully connected pour transformer en vecteur fixe
        self.fc = nn.Linear(style_embed_dim, output_dim)
        
    def forward(self, style_indices):
        # style_indices : [batch_size, style_seq_len] (ex : plusieurs mots de style)
        x = self.embedding(style_indices)  # [batch_size, style_seq_len, style_embed_dim]
        x = x.mean(dim=1)                  # moyenne sur les mots du style [batch_size, style_embed_dim]
        style_vector = self.fc(x)          # [batch_size, output_dim]
        return style_vector

# === 3. Part-based Attention Block ===
class PartAttentionBlock(nn.Module):
    def __init__(self, text_dim=256, feature_dim=128):
        super(PartAttentionBlock, self).__init__()
        # FC pour calculer un poids d'attention à partir du texte de la partie
        self.fc_text = nn.Linear(text_dim, feature_dim)
        self.fc_feat = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, part_text_feature, image_feature):
        """
        part_text_feature : vecteur de la partie (ex : "jupe longue en wax") [batch, text_dim]
        image_feature : caractéristique locale de l'image [batch, feature_dim, h, w]
        """
        # Calcul du poids d'attention (entre 0 et 1)
        att_weight = torch.sigmoid(self.fc_text(part_text_feature))  # [batch, feature_dim]
        att_weight = att_weight.unsqueeze(-1).unsqueeze(-1)          # [batch, feature_dim, 1, 1]
        
        # Transformer image_feature pour qu'il ait la bonne taille si besoin
        feat = self.fc_feat(image_feature.permute(0, 2, 3, 1))       # [batch, h, w, feature_dim]
        feat = feat.permute(0, 3, 1, 2)                              # [batch, feature_dim, h, w]
        
        # Appliquer attention : on multiplie image_feature par poids d'attention
        attended_feat = feat * att_weight
        
        return attended_feat

# === 4. Générateur simple avec 3 étapes (G0, G1, G2) ===
class GeneratorStage(nn.Module):
    def __init__(self, input_dim, out_channels):
        super(GeneratorStage, self).__init__()
        # FC pour transformer input en carte de features
        self.fc = nn.Linear(input_dim, 128 * 8 * 8)
        
        # UpSampling + Conv + BatchNorm + ReLU
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, input_vec, part_features=[]):
        """
        input_vec : [batch, input_dim] = bruit + texte + style concaténés
        part_features : liste de tensors à fusionner avec les features
        """
        batch_size = input_vec.size(0)
        x = self.fc(input_vec)  # [batch, 128*8*8]
        x = x.view(batch_size, 128, 8, 8)  # carte de features
        
        # Fusionner features des parties (attention simplifiée)
        for part_feat in part_features:
            # broadcast pour additionner part_feat (supposé [batch, feature_dim, h, w])
            x = x + part_feat
        
        x = self.upsample(x)
        return x

# === 5. Générateur complet ===
class FullGenerator(nn.Module):
    def __init__(self, noise_dim, text_dim, style_dim):
        super(FullGenerator, self).__init__()
        self.stage0 = GeneratorStage(noise_dim + text_dim + style_dim, 64)
        self.stage1 = GeneratorStage(noise_dim + text_dim + style_dim, 32)
        self.stage2 = GeneratorStage(noise_dim + text_dim + style_dim, 3)  # 3 canaux RGB
    
    def forward(self, noise, sentence_feat, style_feat, part_feats):
        """
        noise : bruit aléatoire [batch, noise_dim]
        sentence_feat : vecteur phrase [batch, text_dim]
        style_feat : vecteur style [batch, style_dim]
        part_feats : liste de features d'attention par partie (ex: haut, bas, manches)
        """
        combined = torch.cat([noise, sentence_feat, style_feat], dim=1)  # concaténation
        
        x0 = self.stage0(combined, part_feats)
        x1 = self.stage1(combined, part_feats)
        x2 = self.stage2(combined, part_feats)
        
        # Sortie finale : image 3 canaux
        return torch.tanh(x2)  # valeurs entre -1 et 1

# === 6. Discriminateur simple ===
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1)

# === 7. Exemple d'usage simple ===
if __name__ == "__main__":
    batch_size = 4
    vocab_size = 5000     # taille vocabulaire texte
    style_vocab_size = 20 # taille vocabulaire style
    noise_dim = 100
    
    # Création des modèles
    text_encoder = TextEncoder(vocab_size)
    style_encoder = StyleEncoder(style_vocab_size)
    part_attention = PartAttentionBlock()
    generator = FullGenerator(noise_dim, 256, 256)
    discriminator = Discriminator()
    
    # Exemples d'entrées aléatoires (indices des mots)
    text_input = torch.randint(0, vocab_size, (batch_size, 15))      # 15 mots par phrase
    style_input = torch.randint(0, style_vocab_size, (batch_size, 5)) # 5 mots style max
    
    # Encodage texte et style
    sent_feat = text_encoder(text_input)    # [batch, 256]
    style_feat = style_encoder(style_input) # [batch, 256]
    
    # Générer features pour 2 parties (exemple)
    part_text1 = sent_feat  # Pour exemple, on utilise sent_feat (tu devras extraire par partie)
    part_text2 = sent_feat
    part_feat1 = part_attention(part_text1, torch.randn(batch_size, 128, 16, 16))
    part_feat2 = part_attention(part_text2, torch.randn(batch_size, 128, 16, 16))
    
    part_feats = [part_feat1, part_feat2]
    
    # Générer bruit
    noise = torch.randn(batch_size, noise_dim)
    
    # Génération image
    fake_img = generator(noise, sent_feat, style_feat, part_feats)
    
    print("Image générée forme :", fake_img.shape)  # devrais voir [batch, 3, 64, 64] ou similaire
