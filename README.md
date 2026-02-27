# Welcome to the XunkO repository!

The XunkO Python package implements the **Anakizer** BPE tokenizer. This implementation is **from scratch** with **no external dependencies** and was born as a learning project. As such, the main concerns when creating the code were readibility, doing everything by hand and, in all honesty, having fun while learning about this topic. Efficiency thus took a back seat so, if you are using it, first and foremost thank you and, secondly, be a little patient!

> **Anakizer** is a mix of the galician word *anaco*, which means part, and *tokenizer*.

## Installation process

To install the package simply write in your terminal:

```text
python -m pip install git+https://github.com/EBoquete02/XunkO.git
```

and you are done! Simply add:

```python
from xunko import Anakizer
```

to start using the tokenizer! Have fun!

## Introduction to Byte Pair Encoding (BPE) tokenization

Tokenization is the first step in any Deep Learning (DL) model that works with text data, such as the Large Language Models (LLMs) that took the world by storm. The whole tokenization problem aims to answer the question:

**How can we represent text data numerically, the only thing that a DL model truly understands, in an efficient way?**

At first we might think that we can simply assign each character with a number and process the text character by character, and in fact this is how text in computers work! Each character is assigned a unique number according to an *encoding*, basically a lookup table that records the character - number pairs. This falls short, however, because tokenizing text would need an incredible amount of numbers, one per character, which would be very costly when they are passed to the DL model waiting down the pipeline.

We might then think to assign a number to each word but, in all honesty, this is incredibly tedious work and would break if the user makes a spelling mistake or uses a word from a different language.

This is why BPE tokenization was invented, it is a middle ground between these two approaches: it isn't character level, so tokenized text is not as long, but it is also universal and saves us from going over all possible words by hand. It works by assigning unique numbers not to characters nor words, but to text chunks called **tokens** that the tokenizer learns by itself.

## How does the tokenizer learn?
You heard it right, the algorithm itself chooses what text pieces are worth numbering without any input from the user apart from a training text. This is not magic, however, it is just the result of the following training loop:

1. The tokenizer takes all unique characters from the text and assigns them a number, thus converting them into tokens. In the **Anakizer** the correspondances between tokens and their **ids** are stored in the attributes **Anakizer.token_to_id** and **Anakizer.id_to_token**.

2. The text gets tokenized with the current **vocabulary**, which is simply the name given to the tokens that the tokenizer knows or, in even plainer english, the text chunks to which the tokenizer assigned numbers. Since our tokenizer only knows thus far individual characters, we will tokenize the training text on a character level.

3. *This is where the fun begins.* At this stage we enter the actual training loop, where the tokenizer learns new tokens by **merging the most common consecutive pairs**, but let us be a little more explicit:

    - The tokenizer goes over all consecutive token id pairs in the tokenized text and grabs the most frequent. For example, given a tokenized text like [1, 3, 4, 1, 3] the pairs would be (1, 3), (3, 4), (4, 1) and (1, 3), making (1, 3) the most common one as it appears twice and the rest only once.

    - The two token ids get detokenized back into text, and the token is created by simply concatenating them. In our previous example, imagine that 'a' is the token that corresponds to 1 and 'c' the one that corresponds to 3, then the new token is 'ac'.

    - The new token is assigned a new token id, by taking the higher token id and adding one. If in our example the tokenizer already knows 100 tokens, it would assing 101 to our new 'ac' token.

    - Every instance of the selected pair in the tokenized text is substituted by its new id to obtain the new tokenization of the training text, and the loop begins once again. The loop finishes when all pairs appear only once or when the vocabulary size reaches the number the user desires.

The **Anakizer** implements this algorithm in its **train** method. It takes two inputs, the training text as a string and the integer containing the desired vocabulary size.

## Implementation details: Anakizer public API

The **Anakizer** implements five public methods:

1. The **train** method implements the learning algorithm of the previous section. As we said, it has two inputs: the training text as a string and the integer specifying the vocabulary size.

2. The **encode** method tokenizes the input text. It has two inputs: the text we want to tokenize as a string and an optional boolean argument called add_bos_eos, which lets you choose if the special beginning of sequence (bos) and end of sequence (eos) tokens are included in the tokenization. If the text has a character not present in the vocabulary it will substitute it by the unknown special token id.

3. The **decode** method is the inverse of the **encode** method, and is responsible for converting a list of integers, assumed to be token ids, back into text. Its arguments are the list of token ids and an optional boolean argument skip_specials, which lets you choose if the special tokens present in the list should appear in the detokenized text.

4. The **save** method simply stores the tokenizer state, conformed by the Anakizer attributes: token_to_id, id_to_token, id_pairs_to_id, unk_token, bos_token, eos_token and special_tokens, into a text file at the path specified in its unique argument.

5. The **load** method can load the tokenizer state contained in a file created via the **save** method. Since I wrote everything from scratch the code is only guaranteed to work if the file passed was created with the save method and wasn't later modified, so beware!

## Usage example

Here we show an example of the **Anakizer** in action taken from the test.ipynb notebook of the tests directory. We first initialize an empty tokenizer with the default special tokens and we train them on the most famous Spanish book **El ingenioso hidalgo Don Quijote de la Mancha** with a vocabulary of 200 tokens. Once trained we use it to tokenize and detokenize a few sentences from the book, before saving and loading its state to check that that the process works.

```python
from xunko import Anakizer

tokenizer = Anakizer(unk = '<|unk|>', bos = '<|bos|>', eos = '<|eos|>')
print(f'Say hello to our tokenizer!\n{tokenizer}\n\n')

with open('quijote.txt', mode = 'rt', encoding = 'utf-8') as file:
    train_text = file.read()
tokenizer.train(text = train_text, vocab_size = 200)
print(f'Vocabulary of the tokenizer after training with \'El Quijote\': \n{tokenizer.token_to_id}\n\n')

sentences = [
    '<|bos|>En un lugar de la Mancha, de cuyo nombre no quiero acordarme.<|eos|>',
    '¡Qué locura es esta! ¿Acaso no veis que son gigantes y no molinos?',
    'La libertad, Sancho, es uno de los más preciosos dones que a los hombres dieron los cielos.\t'
]
for sentence in sentences: 
    tokenized_sentence = tokenizer.encode(sentence)
    detokenized_sentence = tokenizer.decode(tokenized_sentence, skip_specials = False)
    print(f'The sentence \'{sentence}\' tokenized looks like : \n{tokenized_sentence}')
    print(f'The reconstruction of the sentence by the tokenizer is : \n{detokenized_sentence}\n')
print('If you do not want to see the special characters present in the original sentences just use skip_specials = True\n\n')

(special_tokens, token_to_id, id_pairs_to_id) = (
    tokenizer.special_tokens.copy(),
    tokenizer.token_to_id.copy(), 
    tokenizer.id_pairs_to_id.copy()
)
tokenizer.save('quijote_save.txt')
tokenizer.load('quijote_save.txt')

print(f'Are the special tokens the same after loadig: {tokenizer.special_tokens == special_tokens}')
print(f'Are the vocabularies the same after loading: {tokenizer.token_to_id == token_to_id}')
print(f'Are the merging rules the same after loading: {tokenizer.id_pairs_to_id == id_pairs_to_id}')
```
```text
Say hello to our tokenizer!
Untrained Anakizer with special tokens: <|unk|>, <|bos|>, <|eos|>


Vocabulary of the tokenizer after training with 'El Quijote': 
{'<|unk|>': 0, '<|bos|>': 1, '<|eos|>': 2, 'E': 3, 'l': 4, ' ': 5, 'i': 6, 'n': 7, 'g': 8, 'e': 9, 'o': 10, 's': 11, 'h': 12, 'd': 13, 'a': 14, 'Q': 15, 'u': 16, 'j': 17, 't': 18, 'M': 19, 'c': 20, '\n': 21, 'T': 22, 'A': 23, 'S': 24, 'Y': 25, ',': 26, 'J': 27, 'G': 28, 'r': 29, 'b': 30, 'C': 31, 'á': 32, 'm': 33, 'R': 34, 'y': 35, 'ñ': 36, 'q': 37, 'f': 38, 'v': 39, 'p': 40, 'é': 41, 'í': 42, ';': 43, '.': 44, 'V': 45, 'I': 46, 'O': 47, 'N': 48, 'D': 49, 'L': 50, 'ó': 51, 'U': 52, '1': 53, '6': 54, '0': 55, '4': 56, 'F': 57, 'P': 58, 'ú': 59, 'z': 60, ':': 61, 'B': 62, 'É': 63, 'x': 64, 'Ó': 65, '¿': 66, '?': 67, '-': 68, '¡': 69, '!': 70, 'X': 71, 'Z': 72, 'ü': 73, '»': 74, 'H': 75, "'": 76, 'Á': 77, 'Í': 78, 'Ñ': 79, '«': 80, '(': 81, ')': 82, '"': 83, 'ï': 84, 'Ú': 85, 'W': 86, ']': 87, 'à': 88, '7': 89, '5': 90, '2': 91, '3': 92, 'ù': 93, 'e ': 94, 'a ': 95, 'o ': 96, 's ': 97, ', ': 98, 'en': 99, 'qu': 100, 'er': 101, 'es': 102, 'an': 103, 'que ': 104, 'de ': 105, 'on': 106, 'ar': 107, 'el': 108, 'y ': 109, 'or': 110, 'os ': 111, 'al': 112, 'ad': 113, 'la ': 114, 'en ': 115, 'os': 116, 'as ': 117, 'el ': 118, 'ab': 119, 'as': 120, 'di': 121, 'o, ': 122, 'est': 123, 'ci': 124, 'on ': 125, 'un': 126, 'ent': 127, 'es ': 128, 'a, ': 129, 'in': 130, 'su': 131, 'ch': 132, 'cu': 133, 'do ': 134, 'mi': 135, 'or ': 136, '\n\n': 137, 're': 138, 'om': 139, 'no ': 140, 'de': 141, 'se ': 142, 'los ': 143, 'tr': 144, 'ó ': 145, 'am': 146, 'le ': 147, '; ': 148, 'lo ': 149, 'ant': 150, 'vi': 151, 'an ': 152, 'e, ': 153, 'si': 154, 'er ': 155, 'ar ': 156, 'se': 157, 'a\n': 158, '. ': 159, 'ti': 160, 'con ': 161, 'par': 162, 'las ': 163, 'com': 164, 'ía ': 165, 'por ': 166, '.\n\n': 167, 'n ': 168, 'to': 169, 'ri': 170, 'pu': 171, 'al ': 172, 'le': 173, 'con': 174, 'ui': 175, 'ot': 176, 'su ': 177, 'des': 178, 'o\n': 179, 'me ': 180, ',\n': 181, 'o de ': 182, 'ed': 183, 'ac': 184, 'a de ': 185, 'all': 186, 'os, ': 187, 'qui': 188, 'br': 189, 'mu': 190, 'ol': 191, 'esp': 192, 'bi': 193, 'anch': 194, 'hab': 195, 'e\n': 196, 'don ': 197, 'Qui': 198, 'pr': 199}


The sentence '<|bos|>En un lugar de la Mancha, de cuyo nombre no quiero acordarme.<|eos|>' tokenized looks like : 
[1, 1, 3, 168, 126, 5, 4, 16, 8, 156, 105, 114, 19, 194, 129, 105, 133, 35, 96, 7, 139, 189, 94, 140, 188, 101, 96, 184, 110, 13, 107, 33, 9, 44, 2, 2]
The reconstruction of the sentence by the tokenizer is : 
<|bos|><|bos|>En un lugar de la Mancha, de cuyo nombre no quiero acordarme.<|eos|><|eos|>

The sentence '¡Qué locura es esta! ¿Acaso no veis que son gigantes y no molinos?' tokenized looks like : 
[1, 69, 15, 16, 41, 5, 4, 10, 133, 29, 95, 128, 123, 14, 70, 5, 66, 23, 20, 120, 96, 140, 39, 9, 6, 97, 104, 11, 125, 8, 6, 8, 150, 128, 109, 140, 33, 191, 130, 116, 67, 2]
The reconstruction of the sentence by the tokenizer is : 
<|bos|>¡Qué locura es esta! ¿Acaso no veis que son gigantes y no molinos?<|eos|>

The sentence 'La libertad, Sancho, es uno de los más preciosos dones que a los hombres dieron los cielos.	' tokenized looks like : 
[1, 50, 95, 4, 6, 30, 101, 18, 113, 98, 24, 194, 122, 128, 126, 182, 143, 33, 32, 97, 40, 138, 124, 116, 111, 13, 106, 128, 104, 95, 143, 12, 139, 189, 128, 121, 101, 125, 143, 124, 108, 116, 44, 0, 2]
The reconstruction of the sentence by the tokenizer is : 
<|bos|>La libertad, Sancho, es uno de los más preciosos dones que a los hombres dieron los cielos.<|unk|><|eos|>

If you do not want to see the special characters present in the original sentences just use skip_specials = True


Are the special tokens the same after loadig: True
Are the vocabularies the same after loading: True
Are the merging rules the same after loading: True
```