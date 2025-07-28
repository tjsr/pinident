from boxdata import BoxData


class FrameData:
	__boxes: list[BoxData]

	@property
	def has_pin(self) -> bool:
		return self.__boxes is not None and len(self.__boxes) > 0

	@property
	def pin_count(self) -> int:
		if self.__boxes is None:
			return 0
		return len(self.__boxes)

