import re

def extract_main_body(latex_text):
    """
    Extracts the main body of a LaTeX document between \begin{document} and \end{document}.
    """
    # Use regex to extract text between \begin{document} and \end{document}
    body_match = re.search(r'\\begin{document}(.*?)\\end{document}', latex_text, re.DOTALL)
    if body_match:
        return body_match.group(1)  # Return the content within \begin{document} and \end{document}
    return None

def clean_latex_text(latex_text):
    """
    Removes LaTeX comments, and cleans up the LaTeX text to get the plain text content.
    """   
    # Remove comments (lines that start with %)
    cleaned_text = re.sub(r'%.*', '', latex_text)
    
    # Replace multiple newlines and spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

class LatexEnvironmentHandler:
    def __init__(self):
        # First compiler: Find $ or $$ (even if unmatched)
        self.single_dollar_pattern = re.compile(r'(?<!\$)\$(?!\$)')  # Match single $
        self.double_dollar_pattern = re.compile(r'\$\$')  # Match double $$

        # Second compiler: Find \begin{...} and \[
        self.begin_pattern = re.compile(r'\\begin\{(.*?)\}')  # \begin{...}
        self.display_math_begin_pattern = re.compile(r'\\\[')  # \[

        # Third compiler: Find \end{...} and \]
        self.end_pattern = re.compile(r'\\end\{(.*?)\}')  # \end{...}
        self.display_math_end_pattern = re.compile(r'\\\]')  # \]

    def find_dollars(self, text):
        """
        finds all $ or $$ (even if unmatched).
        """
        single_dollars = list(re.finditer(self.single_dollar_pattern, text))
        double_dollars = list(re.finditer(self.double_dollar_pattern, text))
        return single_dollars + double_dollars

    def find_begins(self, text):
        """
        finds all \begin{...} and \[.
        """
        begin_matches = list(re.finditer(self.begin_pattern, text))
        display_math_begins = list(re.finditer(self.display_math_begin_pattern, text))
        return begin_matches + display_math_begins

    def find_ends(self, text):
        """
        finds all \end{...} and \].
        """
        end_matches = list(re.finditer(self.end_pattern, text))
        display_math_ends = list(re.finditer(self.display_math_end_pattern, text))
        return end_matches + display_math_ends

    def find_and_sort_all_matches(self, text):
        """
        find all matches from all compilers and sorted them by their location in text.
        """
        dollar_matches = self.find_dollars(text)
        begin_matches = self.find_begins(text)
        end_matches = self.find_ends(text)

        # combine all matches
        all_matches = dollar_matches + begin_matches + end_matches

        # sort by their start position/location 
        all_matches.sort(key=lambda match: match.start())

        return all_matches
    
    def get_env_labels(self, word):
        """
        get labels of all matches
        """
        labels = []
        for m in self.find_and_sort_all_matches(word):
            labels.append(m.group(0))
        return labels


def chunk_text_by_words(cleaned_text, tokens_per_chunk, token_overlap=0, char_per_token=4):
    """
    chunks the cleaned LaTeX text based on a word count
    """
    words = cleaned_text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        if current_length + word_length / char_per_token <= tokens_per_chunk:
            current_chunk.append(word)
            current_length += word_length / char_per_token
        else:
            chunks.append(' '.join(current_chunk))
            if token_overlap > 0:
                overlap_words, overlap_length = get_overlap_words(current_chunk, token_overlap, char_per_token)
                current_chunk = overlap_words + [word]
                current_length = overlap_length + word_length / char_per_token
            else:
                current_chunk = [word]
                current_length = word_length / char_per_token
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def get_overlap_words(current_chunk, token_overlap, char_per_token):
    """
    helper function for chunk_text_by_words
    """
    overlap_words = []
    total_tokens = 0
    for word in reversed(current_chunk):
        word_length = len(word)
        tokens = word_length / char_per_token
        if total_tokens + tokens <= token_overlap:
            overlap_words.append(word)
            total_tokens += tokens
        else:
            break
    overlap_words.reverse()  # Reverse to maintain the original order
    return overlap_words, total_tokens


def chunk_text_by_words_latex(cleaned_text, tokens_per_chunk, token_overlap=0, environment_sensitive=False, char_per_token=4):
    """
    Chunks the cleaned LaTeX text based on a word count and handles LaTeX environments like \begin{...} and \[...\].
    environment_sensitive: If True, handle environments without splitting them across chunks.
    returns a list of text chunks.
    """

    if environment_sensitive:
        latex_env_handler = LatexEnvironmentHandler()

    words = cleaned_text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    environment_stack = []  # Stack to track environments

    for word in words:
        word_length = len(word)

        if environment_sensitive:
            env_labels = latex_env_handler.get_env_labels(word)

            if env_labels:
                for label in env_labels:
                    #----------
                    # close env
                    #-----------
                    if label.startswith('\\end'):
                        if environment_stack and environment_stack[-1] == label.replace('\\end', '\\begin'):
                            environment_stack.pop() 
                        else:
                            raise ValueError(f"Environment mismatch: trying to close {environment_stack[-1]} but got {label}")
                    elif label == '\\]':
                        if environment_stack and environment_stack[-1] == '\\[':
                            environment_stack.pop() 
                        else:
                            raise ValueError(f"Environment mismatch: trying to close {environment_stack[-1]} but got {label}")
                    elif label in ('$','$$') and environment_stack and environment_stack[-1] == label:  # Handle inline math ($) and display math ($$) environments
                            environment_stack.pop()
                    else:
                        #----------
                        # open env
                        #-----------
                        environment_stack.append(label)
                
                if environment_stack == []:  # if just emptied, word has the closing env, so add the word, and go to the next word
                    current_chunk.append(word)
                    continue

            if environment_stack: 
                current_chunk.append(word)
                continue
                      
        # check length
        if  current_length + word_length / char_per_token <= tokens_per_chunk:
            current_chunk.append(word)
            current_length += word_length / char_per_token
        else:
            # finalize the current chunk
            chunks.append(' '.join(current_chunk))
            if token_overlap > 0:
                # token overlap
                overlap_words, overlap_length = get_overlap_words_latex(current_chunk, token_overlap, environment_sensitive, char_per_token)
                current_chunk = overlap_words + [word]
                current_length = overlap_length + word_length / char_per_token
            else:
                # reset current chunk with the new word
                current_chunk = [word]
                current_length = word_length / char_per_token

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def get_overlap_words_latex(current_chunk, token_overlap, environment_sensitive=False, char_per_token=4):
    """
    helper function for chunk_text_by_words_latex
    """
    if environment_sensitive:
        latex_env_handler = LatexEnvironmentHandler()

    overlap_words = []
    total_tokens = 0
    environment_stack = []  # Stack to track environments
    
    # start from the end
    for word in reversed(current_chunk):
        word_length = len(word)

        if environment_sensitive:
            env_labels = latex_env_handler.get_env_labels(word)

            if env_labels:
                for label in env_labels:
                    # close env
                    if label.startswith('\\begin'):  # going in reverse
                        if environment_stack and environment_stack[-1] == label.replace('\\begin', '\\end'):
                            # print(f"going backward, trying to close {environment_stack[-1]} and got {label}")
                            environment_stack.pop()
                        else:
                            raise ValueError(f"Environment mismatch: going backward, trying to close {environment_stack[-1]} but got {label}")
                    elif label == '\\[': # going in reverse
                        if environment_stack and environment_stack[-1] == '\\]':
                            environment_stack.pop()  # Properly pop the matching environment
                        else:
                            raise ValueError(f"Environment mismatch: going backward, trying to close {environment_stack[-1]} but got {label}")
                    elif label in ('$','$$') and environment_stack and environment_stack[-1] == label:
                            # Handle inline math ($) and display math ($$) environments
                            environment_stack.pop()
                    else:
                        # open env
                        environment_stack.append(label)
                
                if environment_stack == []:  # if just emptied, word had closing env, add word, and go to the next word
                    overlap_words.append(word)
                    continue

            if environment_stack: 
                overlap_words.append(word)
                continue

        if total_tokens + word_length / char_per_token <= token_overlap:
            overlap_words.append(word)
            total_tokens += word_length / char_per_token
        else:
            break
    overlap_words.reverse()  # Reverse to maintain the original order
    return overlap_words, total_tokens


def process_latex_files(file_paths, tokens_per_chunk=200, token_overlap=20, environment_sensitive=True):

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    cum_text = ''
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as tex_file:
            latex_content = tex_file.read()

        # extract the main body of the LaTeX document
        main_body = extract_main_body(latex_content)

        if main_body is None:
            raise ValueError("Could not find the main body of the document.")

        # clean the LaTeX text to remove comments and extra spaces
        cleaned_text = clean_latex_text(main_body)

        cum_text += cleaned_text + '\n'

    # chunk the cleaned text by word count, handling environments if needed
    chunks = chunk_text_by_words_latex(cum_text, tokens_per_chunk, token_overlap, environment_sensitive=environment_sensitive)

    return chunks
