import torch
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm

class BaseDataset(torch.utils.data.Dataset):
	def __init__(self,**kwargs):
		super(BaseDataset,self).__init__()
		self.text=kwargs.pop('data',[])
		self.tokens=self.text2wordlist(self.text)
		self.word2idx={word:i for i,word in enumerate(set(self.tokens))}
		self.idx2word_table={i:word for i,word in enumerate(set(self.tokens))}
		self.inputs=[([self.word2idx[self.tokens[i]],self.word2idx[self.tokens[i+1]],self.word2idx[self.tokens[i+2]]],
					[self.word2idx[self.tokens[i+1]],self.word2idx[self.tokens[i+2]],self.word2idx[self.tokens[i+3]]]) for i in range(len(self.tokens)-3)]
		self.inputs=[(torch.tensor(x),torch.tensor(y)) for x,y in self.inputs]
	def __len__(self):
		return len(self.inputs)
	def __getitem__(self,index):
		return self.inputs[index][0], self.inputs[index][1]
	def get_word2index_map(self):
		return self.word2idx
	def idx2word(self,index):
		return self.idx2word_table[index]
	def get_token_num(self):
		return len(list(set(self.tokens)))
	def text2wordlist(self,text):
		stop_words=set(stopwords.words('english'))
		cut_tokens = nltk.wordpunct_tokenize(text)
		tokens=list()
		stop_words.add('.')
		stop_words.add('?')
		stop_words.add(')')
		stop_words.add(').')
		stop_words.add('(')
		stop_words.add('.(')
		stop_words.add(',')
		stop_words.add('/')
		stop_words.add('-')
		stop_words.add('_')
		stop_words.add('+')
		stop_words.add('$')
		stop_words.add('&')
		stop_words.add('!')
		stop_words.add(';')
		stop_words.add('\'')
		stop_words.add(':')
		# print(stop_words)
		for i in cut_tokens:
			if i not in stop_words:
				tokens.append(i)
		return tokens
	# collate_fn用在將多筆資料放同個batch後對batch內容的處理，這裡用不到
	# def collate_fn(self,batch):


class NGramRNN(torch.nn.Module):
	def __init__(self,vocab_size,output_dim,N):
		super(NGramRNN,self).__init__()
		self.embedding=torch.nn.Embedding(num_embeddings=vocab_size,
							embedding_dim=vocab_size)
		self.embedding.weight.requires_grad=False
		self.rnn_layer = torch.nn.RNN(input_size=vocab_size,
                                      hidden_size=output_dim,
                                      num_layers=1,
                                      nonlinearity='relu',
                                      dropout=0,
                                      batch_first=True)
		self.linear=torch.nn.Linear(output_dim,vocab_size)
	def forward(self,inputs):
		embed=self.embedding(inputs)
		RNN_out,_=self.rnn_layer(embed)
		linear_out=self.linear(RNN_out)
		return linear_out
	def test(self,inputs):
		embed=self.embedding(inputs)
		_,RNN_out=self.rnn_layer(embed)
		linear_out=self.linear(RNN_out)
		return linear_out
	def generate(self,inputs):
		with torch.no_grad():
			embed=self.embedding(inputs)
			_,RNN_out=self.rnn_layer(embed)
			return RNN_out
def syntatic_test():
	cosine_value=cosine(a,b)
	return sum
def semantic_test():
	cosine_value=cosine(a,b)
	return sum


# 以下為測試資料
# data="""When forty winters shall besiege thy brow,
# And dig deep trenches in thy beauty's field,
# Thy youth's proud livery so gazed on now,
# Will be a totter'd weed of small worth held:
# Then being asked, where all thy beauty lies,
# Where all the treasure of thy lusty days;
# To say, within thine own deep sunken eyes,
# Were an all-eating shame, and thriftless praise.
# How much more praise deserv'd thy beauty's use,
# If thou couldst answer 'This fair child of mine
# Shall sum my count, and make my old excuse,'
# Proving his beauty by succession thine!
# This were to be new made when thou art old,
# And see thy blood warm when thou feel'st it cold."""

f=open("data2.txt","r",encoding="utf8")
data=f.read()
f.close()
dataset=BaseDataset(data=data)
print(dataset.get_token_num())
data_loader =torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)
# print(word2idx)
word_embed_dim=800
N=4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_num=dataset.get_token_num()
model=NGramRNN(vocab_num,word_embed_dim,N)
#criterion=torch.nn.NLLLoss()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

model=model.to(device)
#訓練
for epoch in range(100):
	for x,y in tqdm(data_loader,desc='training'):
		model.zero_grad()
		x=x.to(device)
		y=y.to(device)
		output=model(x)
		output=output.view(-1,vocab_num)
		y=y.view(-1)

		loss=criterion(output, y)
		loss.backward()
		optimizer.step()
	print(epoch)
#test有沒有準確預測下一個字
data_test =torch.utils.data.DataLoader(dataset,batch_size=1)
for x,y in data_test:
	with torch.no_grad():
		x=x.to(device)
		output=model.test(x)
		output=output.view(-1)
		output=torch.nn.functional.softmax(output,dim=0)
		output=output.tolist()
		# print(output)
		print([dataset.idx2word(i) for i in x.view(-1).tolist()] ,end=' ')
		print(dataset.idx2word(output.index(max(output))))
#取出訓練的向量
for x,y in data_test:
	x=x.to(device)
	output=model.generate(x)
	output=output.view(-1).tolist()
	print(dataset.idx2word(y.view(-1).tolist()[-1]),end=' ')
	print(output)
	
torch.save(model.state_dict(),'model.ckpt')