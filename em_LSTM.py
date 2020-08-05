import torch
from nltk.corpus import stopwords
import nltk

class NGramRNN(torch.nn.Module):
	def __init__(self,vocab_size,output_dim,N):
		super(NGramRNN,self).__init__()
		self.embedding=torch.nn.Embedding(num_embeddings=vocab_size,
							embedding_dim=vocab_size)
		self.embedding.weight.requires_grad=False
		self.rnn_layer = torch.nn.LSTM(input_size=vocab_size,
                                      hidden_size=output_dim,
                                      num_layers=1,
                                      dropout=0,
                                      batch_first=True,
                                      bidirectional=False)
		self.linear=torch.nn.Linear(output_dim,vocab_size)
	def forward(self,inputs):
		embed=self.embedding(inputs)
		RNN_out,ht=self.rnn_layer(embed)
		linear_out=self.linear(RNN_out)
		return linear_out

	#測試是否成功預測下個字
	def test(self,inputs):
		embed=self.embedding(inputs)
		_,(RNN_out,cell_out)=self.rnn_layer(embed)

		linear_out=self.linear(RNN_out)
		return linear_out

	#將輸入的字轉為詞向量
	def generate(self,inputs):
		with torch.no_grad():
			embed=self.embedding(inputs)
			_,(RNN_out,cell_out)=self.rnn_layer(embed)
			return RNN_out
stop_words=set(stopwords.words('english'))
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


input_data=[([tokens[i],tokens[i+1],tokens[i+2]],[tokens[i+1],tokens[i+2],tokens[i+3]]) for i in range(len(tokens)-3)]
tokens=set(tokens)
tokens=list(tokens)
word2idx={word:i for i,word in enumerate(tokens)}

print(len(tokens))

model=NGramRNN(len(tokens),10,4)
#criterion=torch.nn.NLLLoss()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
#訓練
for epoch in range(100):
	for x,y in input_data:
		input_idxs=torch.tensor([[word2idx[w] for w in x]])
		model.zero_grad()
		output=model(input_idxs)
		# print(output)
		target_y = [word2idx[w] for w in y]
		output = output.view(-1, len(tokens))
		loss=criterion(output, torch.tensor(target_y))
		loss.backward()
		optimizer.step()
		# break
	print(epoch)
	# break
#test有沒有準確預測下一個字
for x,y in input_data:
	with torch.no_grad():
		input_idxs=torch.tensor([[word2idx[w] for w in x]])
		output=model.test(input_idxs)
		output=output.view(-1)
		output=torch.nn.functional.softmax(output)
		output=output.tolist()
		# print(output)
		print(x,end=' ')
		print(tokens[output.index(max(output))])
		print((max(output)))	
#取出訓練的向量
for x,y in input_data:
	input_idxs=torch.tensor([[word2idx[w] for w in x]])
	output=model.generate(input_idxs)
	output=output.view(-1).tolist()
	print(y[-1],end=' ')
	print(output)
	
#torch.save(model.state_dict(),'model.ckpt')