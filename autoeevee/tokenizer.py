
UNKNOWN_TOKEN: int = -1


class ShowdownTokenizer:
    def __init__(self):
        self._initial_ids: dict[str, int] = {}
        self._new_ids: dict[str, int] = {}
        self._frozen: bool = True
        self.name: str = "custom"

    def unfreeze(self):
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def __len__(self):
        return len(self._initial_ids.keys()) + len(self._new_ids.keys())

    @property
    def all_words(self) -> list[str]:
        return list(self._initial_ids.keys()) + list(self._new_ids.keys())

    @property
    def new_token(self):
        return len(self)

    def __getitem__(self, string: str) -> int:
        if string in self._initial_ids:
            return self._initial_ids[string]
        if string in self._new_ids:
            return self._new_ids[string]
        return UNKNOWN_TOKEN

    def save_tokens_to_disk(self, path):
        with open(path, "w") as f:
            json.dump({**self._initial_ids, **self._new_ids}, f)

    def load_tokens_from_disk(self, path):
        with open(path, "r") as f:
            self._initial_ids = json.load(f)
        return self

    def load_tokens(self, tokens: dict[str, int]):
        self._initial_ids = tokens
        return self

    def add_token_for(self, string: str) -> None:
        if string in self._initial_ids:
            return
        if string in self._new_ids:
            return
        print(f"Adding: `{string}`")
        self._new_ids[string] = self.new_token

    def sort_tokens(self) -> None:
        self._new_ids = {
            k: i + len(self._initial_ids)
            for i, k in enumerate(sorted(self._new_ids.keys()))
        }

    def tokenize(self, text: str) -> np.ndarray:
        words = text.split(" ")
        if not self._frozen:
            for word in words:
                self.add_token_for(word)
        return np.array([self[word] for word in words], dtype=np.int32)

