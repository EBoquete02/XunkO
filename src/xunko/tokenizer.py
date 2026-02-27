from collections.abc import Iterable, Hashable

class Anakizer:
    '''
    A Byte Pair Encoding based tokenizer with
    support for special tokens.

    :type unk: str | None
    :param unk: Special token for representing
                unknown characters. It has a
                default value of '<|unk|>'
    :type bos: str | None
    :param bos: Special token that marks the
                beginning of a sequence. It
                has a default value of '<|bos|>'
    :type eos: str | None
    :param eos: Special token that marks the
                end of a sequence. It has a
                default value of '<|eos|>'
    :type specials: str
    :param specials: Additional user defined
                     special tokens. These are
                     treated as atomic units
                     and are protected from being
                     split during the BPE process

    '''

    def __init__(self,
            unk: str | None = '<|unk|>',
            bos: str | None = '<|bos|>',
            eos: str | None = '<|eos|>',
            *specials: str
            ) -> None:

        self.token_to_id = {}
        self.id_to_token = {}
        self.id_pairs_to_id = {}

        self.unk_token = unk
        self.bos_token = bos
        self.eos_token = eos

        special_tokens = []
        for special_token in (unk, bos, eos) + specials:
            if special_token is not None:
                special_tokens.append(special_token)
        self.special_tokens = self.__originals(special_tokens)

    def __str__(self) -> str:
        
        status = 'Trained' if self.token_to_id else 'Untrained'
        special_tokens = ', '.join(self.special_tokens)
        
        return f'{status} Anakizer with special tokens: {special_tokens}'
        
    def __repr__(self) -> str:

        return self.__str__()

    def train(self, text: str, vocab_size: int) -> None:
        '''
        This method trains the BPE tokenizer
        by creating new tokens with the most
        common consecutive pairs in the text
        until reaching the specified vocabulary
        size or no more mergings can be done.

        Note that this method resets any
        existing vocabulary when called.

        Note that the user can specify a
        vocabulary size smaller than the
        number of special tokens given in
        the constructor. In that case the
        tokenizer will prioritize the 
        vocabulary size and thus skip the
        remaining special tokens.
        
        :type text: str
        :param text: Training text
        :type vocab_size: int
        :param vocab_size: Maximum number of
                           tokens in the
                           vocabulary

        :rtype: None
        :return: This method updates the 
                 tokenizer vocabulary in
                 place with no return

        :example:
            >>> tokenizer = Anakizer('<|unk|>', None, None)
            >>> train_text = 'aaabaaac'
            >>> tokenizer.train(text = train_text, vocab_size = 6)
            >>> tokenizer.token_to_id
            {'<|unk|>': 0, 'a': 1, 'b': 2, 'c': 3, 'aa': 4, 'aaa': 5}

        '''

        self.token_to_id = {}
        self.id_to_token = {}
        self.id_pairs_to_id = {}
        
        chars = self.__originals(text)
        chars = self.__originals(self.special_tokens + chars)

        new_token_id = 0
        for token in chars:
            if new_token_id >= vocab_size:
                break
            self.token_to_id[token] = new_token_id
            self.id_to_token[new_token_id] = token
            new_token_id = new_token_id + 1

        token_ids = self.encode(text, add_eos_bos = False)

        while new_token_id < vocab_size:

            (best_pair, best_freq) = self.__cupid(token_ids)
            if best_pair is None or best_freq == 1:
                break
            token_ids = self.__marriage(token_ids, best_pair, new_token_id)

            (lead_id, follow_id) = best_pair
            lead_token = self.id_to_token[lead_id]
            follow_token = self.id_to_token[follow_id]
            new_token = lead_token + follow_token

            self.token_to_id[new_token] = new_token_id
            self.id_to_token[new_token_id] = new_token
            self.id_pairs_to_id[best_pair] = new_token_id

            new_token_id = new_token_id + 1

    def encode(self, text: str, add_eos_bos: bool = True) -> list[int]:
        '''
        This method tokenizes the input
        text into a list of token ids. It
        first tokenizes on a character and
        special token level before applying
        the merging rules on the order 
        learned during training.

        If there is an unknown character in text
        the method will use the unknown token
        in its place. If the unknown token is 
        not in the vocabulary it will instead
        skip the character and print a warning
        indicating it.
        
        :type text: str
        :param text: The text we wish to encode
        :type add_eos_bos: bool
        :param add_eos_bos: If True the special 
                            bos and eos tokens 
                            will be added at the
                            beginning and end of 
                            a sequence provided 
                            they are present in
                            the vocabulary

        :rtype: list[int]
        :return: The tokenized version of text

        :raises RuntimeError: If the vocabulary is 
                              empty. Use the train
                              or load methods first

        :examples:
            >>> token = Anakizer('<|unk|>', '<|bos|>', '<|eos|>')
            >>> token.train('Hello world', 10)
            >>> token.encode('Hello world!')
            [1, 3, 4, 5, 5, 6, 7, 8, 6, 9, 5, 10, 0, 2]
            >>> token.encode('Hello world!', add_eos_bos = False)
            [3, 4, 5, 5, 6, 7, 8, 6, 9, 5, 10, 0]

        '''
        if not self.token_to_id:
            raise RuntimeError('Trying to encode with an empty vocabulary')

        encoding = []
        remainder = text
        unknown_chars = []

        while remainder:

            sep = False
            min_pos = None

            for special_token in self.special_tokens:
                try:
                    pos = remainder.index(special_token)
                except ValueError:
                    continue
                
                if min_pos is None or pos < min_pos:
                    sep = special_token
                    min_pos = pos
                elif pos == min_pos and sep in special_token:
                    sep = special_token

            if sep:
                enc, sep, remainder = remainder.partition(sep)
            else:
                enc, sep, remainder = remainder, '', ''

            for char in enc:
                self.__safe_append(char, encoding, unknown_chars)
            if sep:
                self.__safe_append(sep, encoding, unknown_chars)

        if unknown_chars:
            unknown_chars = self.__originals(unknown_chars)
            print(f'WARNING: Skipping unknown characters {unknown_chars}')
        
        for id_pair in self.id_pairs_to_id:
            new_id = self.id_pairs_to_id[id_pair]
            encoding = self.__marriage(encoding, id_pair, new_id)

        if add_eos_bos:
            try:
                encoding.insert(0, self.token_to_id[self.bos_token])
            except KeyError:
                pass
            try:
                encoding.append(self.token_to_id[self.eos_token])
            except KeyError:
                pass

        return encoding
    
    def decode(self, token_ids: list[int], skip_specials: bool = False) -> str:
        '''
        This method decodes the list token_ids 
        into a string by concatenating their 
        corresponding tokens.

        If there is an unknown id in the list
        the method will skip it and print a
        warning indicating it.
        
        :type token_ids: list[int]
        :param token_ids: The list of token
                          ids we wish to 
                          decode
        :type skip_specials: bool
        :param skip_specials: If True the special
                              tokens present in
                              token_ids will not be
                              present in the decoded
                              string

        :rtype: str
        :return: The decoded text

        :raises RuntimeError: If the vocabulary is 
                              empty. Use the train
                              or load methods first

        :examples:
            >>> token = Anakizer('<|unk|>', '<|bos|>', '<|eos|>')
            >>> token.train('Hello world', 10)
            >>> token_ids = [1, 3, 4, 5, 5, 6, 7, 8, 6, 9, 5, 10, 0, 2]
            >>> token.decode(token_ids = token_ids)
            <|bos|>Hello world<|unk|><|eos|>
            >>> token.decode(token_ids = token_ids, skip_specials = True)
            Hello world

        '''
        if not self.id_to_token:
            raise RuntimeError('Trying to decode with an empty vocabulary')
        
        decoding = []
        unknown_ids = []

        if skip_specials:
            special_ids = {self.token_to_id[token] for token in self.special_tokens}
        else:
            special_ids = []

        for idx in token_ids:
            if idx not in special_ids:
                try:
                    decoding.append(self.id_to_token[idx])
                except KeyError:
                    unknown_ids.append(idx)
        
        if unknown_ids:
            unknown_ids = self.__originals(unknown_ids)
            print(f'WARNING: Skipping unknown ids {unknown_ids}')

        decoding = ''.join(decoding)

        return decoding
    
    def save(self, path: str) -> None:
        '''
        This method saves the state 
        of the tokenizer into a text 
        file at the specified path.
        
        :type path: str
        :param path: The path where
                     the text file will
                     be saved

        :rtype: None
        :return: This method creates
                 a text file with the
                 state of the tokenizer
                 with no return

        :example:
            >>> tokenizer = Anakizer('<|unk|>', None, None)
            >>> tokenizer.train('aaabaaac', 6)
            >>> tokenizer.save('save.txt')
            >>> with open('save.txt') as file:
            ... file.read()
            <|unk|>
            None
            None
            <|unk|>
            <|unk|> 0
            a       1
            b       2
            c       3
            aa      4
            aaa     5
            (1, 1)  4
            (4, 1)  5

        '''

        with open(path, mode = 'wt', encoding = 'utf-8') as file:
            for token in (self.unk_token, self.bos_token, self.eos_token):
                token = f'{token}'.encode('unicode_escape').decode('utf-8')
                file.write(f'{token}\n')
            special_tokens = []

            for token in self.special_tokens:
                token = f'{token}'.encode('unicode_escape').decode('utf-8')
                special_tokens.append(token)
            special_tokens = '\v'.join(special_tokens)
            file.write(f'{special_tokens}\n')

            file.write(f'{self.__save_vocab(self.token_to_id)}\n')
            file.write(f'{self.__save_vocab(self.id_pairs_to_id)}\n')

    def load(self, path: str) -> None:
        '''
        This method reads the text file
        located at the given path and 
        loads the tokenizer state it 
        contains. The previous state is 
        lost.
        
        :type path: str
        :param path: The text file from 
                     which we load the 
                     tokenizer state

        :rtype: None
        :return: This method replaces the
                 current tokenizer state
                 with the one found in the
                 text file with no return

        :example:
            >>> tokenizer = Anakizer('<|unk|>', None, None)
            >>> tokenizer.train('aaabaaac', 6)
            >>> tokenizer.save('save.txt')
            >>> tokenizer.load('save.txt')
            >>> tokenizer.token_to_id
            {'<|unk|>': 0, 'a': 1, 'b': 2, 'c': 3, 'aa': 4, 'aaa': 5}

        '''

        self.token_to_id = {}
        self.id_to_token = {}
        self.id_pairs_to_id = {}

        with open(path, mode = 'rt', encoding = 'utf-8') as file:

            lines = []
            for line in file:
                lines.append(line.replace('\n', ''))

            unk_token = lines[0].encode('utf-8').decode('unicode_escape')
            self.unk_token = None if unk_token == 'None' else unk_token
            bos_token = lines[1].encode('utf-8').decode('unicode_escape')
            self.bos_token = None if bos_token == 'None' else bos_token
            eos_token = lines[2].encode('utf-8').decode('unicode_escape')
            self.eos_token = None if eos_token == 'None' else eos_token
            
            self.special_tokens = []
            special_tokens = lines[3].split('\v') if lines[3] else []
            for token in special_tokens:
                token = token.encode('utf-8').decode('unicode_escape')
                self.special_tokens.append(token)

            token_to_id = lines[4].split('\v') if lines[4] else []
            for token_id in token_to_id:
                token_id = token_id.split('\t')
                token = token_id[0].encode('utf-8').decode('unicode_escape')
                self.token_to_id[token] = int(token_id[1])
                self.id_to_token[int(token_id[1])] = token

            id_pairs_to_id = lines[5].split('\v') if lines[5] else []
            for id_pair_id in id_pairs_to_id:
                id_pair_id = id_pair_id.split('\t')
                id_pair = id_pair_id[0][1 : -1].split(', ')
                id_pair = (int(id_pair[0]), int(id_pair[1]))

                self.id_pairs_to_id[id_pair] = int(id_pair_id[1])

    def __save_vocab(self, dictionary: dict) -> str:
        '''
        This method serializes a dictionary 
        into a string using \\v to separate
        items and \\t to separate keys from
        values.
        
        :type dictionary: dict
        :param dictionary: The dictionary we wish to
                           turn into a string

        :rtype: str
        :return: The string representation of
                 the dictionary

        :example:
            >>> token = Anakizer('<|unk|>', None, None)
            >>> token.train('aaabaaac', 6)
            >>> token._Anakizer__save_vocab(token.token_to_id)
            <|unk|> 0
            a       1
            b       2
            c       3
            aa      4
            aaa     5

        '''

        items = []

        dictionary_iterator = dictionary.__iter__()

        while True:
            try:
                key = dictionary_iterator.__next__()
            except StopIteration:
                break
            value = f'{dictionary[key]}'
            key = f'{key}'

            key = key.encode('unicode_escape').decode('utf-8')
            value = value.encode('unicode_escape').decode('utf-8')

            item = key + '\t' + value
            items.append(item)

        dictionary_str = '\v'.join(items)

        return dictionary_str

    def __safe_append(self, token: str, encoding: list[int], unknown_chars: list[str]) -> None:
        '''
        This method tries to append the
        token id of token to the list
        encoding. If token is not known it
        will instead try to append the token
        id of the unknown special token. If 
        this also fails it will append token 
        to the list unknown_chars.
        
        :type token: str
        :param token: The token we want to
                      append to encoding
        :type encoding: list[int]
        :param encoding: The list to which
                         we want to append
                         the id of token
        :type unknown_chars: list[str]
        :param unknown_chars: If token is not
                              known and we have
                              no unknown special
                              token the former
                              will be appended
                              here

        :rtype: None
        :return: This method modifies either the
                 encoding or unknown_chars list in
                 place with no return

        :examples:
            >>> token = Anakizer('<|unk|>', None, None)
            >>> token.train('aaabaaac', 6)
            >>> (enc, unk) = ([], [])
            >>> token._Anakizer__safe_append('a', enc, unk)
            >>> enc
            [1]
            >>> token._Anakizer__safe_append('!', enc, unk)
            >>> enc
            [1, 0]
            >>> token.unk_token = None
            >>> token._Anakizer__safe_append('?', enc, unk)
            >>> unk
            ['?']

        '''

        try:
            encoding.append(self.token_to_id[token])
        except KeyError:
            try:
                encoding.append(self.token_to_id[self.unk_token])
            except KeyError:
                unknown_chars.append(token)

    def __originals(self, iterable: Iterable[Hashable]) -> list[Hashable]:
        '''
        This method takes any iterable
        with hashable elements and returns
        a list with said elements appearing
        only once preserving first occurrence
        order.

        :type iterable: Iterable[Hashable]
        :param iterable: An arbitrary iterable of
                         hashable elements

        :rtype: list[Hashable]
        :return: A list with each of the 
                 elements of the iterable
                 appearing only once in
                 order of first appearance

        :example:
            >>> token = Anakizer('<|unk|>', None, None)
            >>> test_iter = 'abcda'
            >>> token._Anakizer__originals(test_iter)
            ['a', 'b', 'c', 'd']

        '''

        seen_elements = {}
        original_elements = []

        iterator = iterable.__iter__()

        while True:
            try:
                elem = iterator.__next__()
            except StopIteration:
                break

            try:
                seen_elements[elem]
            except KeyError:
                original_elements.append(elem)
                seen_elements[elem] = True

        return original_elements

    def __marriage(self, token_ids: Iterable, id_pair: tuple, new_id) -> list:
        '''
        This method takes an arbitrary iterable
        and returns a list containing the same
        elements but with every instance of id_pair
        substituted by new_id.
        
        :type token_ids: Iterable[Any]
        :param token_ids: An arbitrary iterable
        :type id_pair: tuple[Any, Any]
        :param id_pair: The consecutive pair to
                        be substituted
        :type new_id: Any
        :param new_id: The substitution of id_pair

        :rtype: list[Any]
        :return: A list containing the same elements
                 of token_ids with every instance 
                 of id_pair substituted by new_id

        :example:
            >>> token = Anakizer('<|unk|>', None, None)
            >>> test_iter = [1, (4, 'hello'), 2, 3]
            >>> token._Anakizer__marriage(test_iter, (2, 3), 'world')
            [1, (4, 'hello'), 'world']

        '''
        
        new_token_ids = []

        token_ids_iterator = token_ids.__iter__()

        try:
            groom = token_ids_iterator.__next__()
        except StopIteration:
            return new_token_ids
        
        while True:
            try: 
                bride = token_ids_iterator.__next__()
            except StopIteration:
                new_token_ids.append(groom)
                break

            if (groom, bride) == id_pair:
                new_token_ids.append(new_id)
                try:
                    groom = token_ids_iterator.__next__()
                except StopIteration:
                    break
            else:
                new_token_ids.append(groom)
                groom = bride

        return new_token_ids

    def __cupid(self, token_ids: Iterable[Hashable]) -> tuple[tuple[Hashable, Hashable] | None, int]:
        '''
        This method takes any iterable with 
        hashable elements and returns the most 
        frequent consecutive pair along with
        its frequency.

        :type token_ids: Iterable[Hashable]
        :param token_ids: An arbitrary iterable of 
                          hashable elements

        :rtype: tuple[tuple[Hashable, Hashable] | None, int]
        :return: tuple of the most common consecutive
                 pair (tuple) and its frequency (int)

        :example:
            >>> token = Anakizer('<|unk|>', None, None)
            >>> test_iter = [1, 2, 1, 2, 'hello world']
            >>> token._Anakizer__cupid(test_iter)
            ((1, 2), 2)

        '''

        candidates = {}
        best_pair = None
        max_freq = 0

        token_ids_iterator = token_ids.__iter__()

        try:
            male = token_ids_iterator.__next__()
        except StopIteration:
            return (best_pair, max_freq)

        while True:
            try:
                female = token_ids_iterator.__next__()
            except StopIteration:
                break

            pair = (male, female)
            male = female

            try:
                candidates[pair] = candidates[pair] + 1
            except KeyError:
                candidates[pair] = 1

        candidates_iterator = candidates.__iter__()
        
        while True:
            try:
                pair = candidates_iterator.__next__()
            except StopIteration:
                break

            if max_freq < candidates[pair]:
                best_pair = pair
                max_freq = candidates[pair]

        return (best_pair, max_freq)