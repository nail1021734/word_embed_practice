import torch
import nltk
from nltk.corpus import stopwords
class NGramModule(torch.nn.Module):
	def __init__(self,vocab_size,output_dim,N):
		super(NGramModule,self).__init__()
		self.embedding=torch.nn.Embedding(num_embeddings=vocab_size,
							embedding_dim=vocab_size)
		self.embedding.weight.requires_grad=False
		self.linear1=torch.nn.Linear(vocab_size*N,output_dim)
		self.linear2=torch.nn.Linear(output_dim,vocab_size)
	def forward(self,inputs):
		embed=self.embedding(inputs).view((1,-1))
		linear1_out=self.linear1(embed)
		linear1_out=torch.nn.functional.relu(linear1_out)
		linear2_out=self.linear2(linear1_out)
		return linear2_out
	def generate(self,inputs):
		with torch.no_grad():
			output=self.embedding(inputs)
			embed=self.embedding(inputs).view((1,-1))
			linear1_out=self.linear1(embed)
			return linear1_out

stop_words=set(stopwords.words('english'))
# f=open('C:/Users/nail1/Desktop/RNN/language-model-playground-master/data/data.txt','r',encoding='utf8')

# data=f.read()
# f.close()
data="""When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold."""

cut_tokens = nltk.wordpunct_tokenize(data)
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


input_data=[([tokens[i],tokens[i+1],tokens[i+2]],tokens[i+3]) for i in range(len(tokens)-3)]
tokens=set(tokens)
tokens=list(tokens)
word2idx={word:i for i,word in enumerate(tokens)}
#print(word2idx)


# print(input_data)
print(len(tokens))

model=NGramModule(len(tokens),10,3)
#criterion=torch.nn.NLLLoss()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(100):
	for x,y in input_data:
		input_idxs=torch.tensor([word2idx[w] for w in x])
		model.zero_grad()
		output=model(input_idxs)
		loss=criterion(output,torch.tensor([word2idx[y]]))
		loss.backward()
		optimizer.step()
	print(epoch)
#test有沒有準確預測下一個字
for x,y in input_data:
	input_idxs=torch.tensor([word2idx[w] for w in x])
	output=model(input_idxs)
	output=torch.nn.functional.softmax(output)
	output=output.tolist()
	print(len(output[0]))
	print(x,end=' ')
	print(tokens[output[0].index(max(output[0]))])
	print((max(output[0])))
#取出訓練的向量
for x,y in input_data:
	input_idxs=torch.tensor([word2idx[w] for w in x])
	output=model.generate(input_idxs)
	output=output.tolist()
	#print(y,end=' ')
	#print(output[0])
	
#torch.save(model.state_dict(),'model.ckpt')