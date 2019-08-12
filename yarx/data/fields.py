from typing import Dict, Any, List

from allennlp.data import Field, DataArray

from allennlp.data.fields import ListField
from overrides import overrides


class OptionalListField(ListField):
    def __init__(self,
                 field_list: List[Field] = None,
                 empty_field: Field = None) -> None:
        """
        An extension of ListField, which is not easy to be wrapped inside
        another List/Sequence because of poor support of empty lists,
        as it is not easy to infer the type of empty list.
        This implementation takes `empty_field` and pads the list with them.
        """
        self._empty_field = empty_field
        self._empty_list = [empty_field]

        if field_list:
            super().__init__(field_list)
            return
        else:
            field_list = []

        self._field_list = field_list
        super().__init__(self._empty_list)
        self._unpad()

    @overrides
    def sequence_length(self) -> int:
        self._pad()
        res = super().sequence_length()
        self._unpad()
        return res

    def _pad(self):
        self._field_list = self.field_list
        if self.field_list:
            return
        if self._empty_field is None:
            raise ValueError("empty_field has to be not None if the list is empty")
        self.field_list = self._empty_list

    def _unpad(self):
        self._empty_list = self.field_list
        self.field_list = self._field_list

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        self._pad()
        res = super().as_tensor(padding_lengths)
        self._unpad()
        return res

    @overrides
    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:
        self._pad()
        res = super().batch_tensors(tensor_list)
        self._unpad()
        return res

    @overrides
    def empty_field(self):
        sample_field = self.field_list[0] if self.field_list else self._empty_field
        empty_field = sample_field.empty_field()
        return OptionalListField([empty_field], empty_field)

    def __str__(self) -> str:
        self._pad()
        res = 'Optional' + super().__str__()
        self._unpad()
        return res

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        self._pad()
        res = super().get_padding_lengths()
        self._unpad()
        return res

    def append(self, field: Field):
        new_index = len(self.field_list)
        self.field_list.append(field)
        return new_index

    def find(self, value: Any) -> int:
        return self.field_list.index(value)
