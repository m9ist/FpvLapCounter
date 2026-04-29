"""
LM Studio Client (опциональный)
Использует vision-модель через OpenAI-совместимый API LM Studio
для верификации сомнительных детекций ворот.
"""

import base64
import numpy as np
import cv2
from typing import Optional


def frame_to_base64(frame: np.ndarray) -> str:
    """Конвертирует numpy frame в base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


class LMStudioClient:
    """
    Клиент для LM Studio.
    Использует vision-модель для анализа сложных кадров.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = None,  # None = первая доступная
        verbose: bool = False,
    ):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.verbose = verbose

        if model is None:
            self._auto_select_model()

    def _auto_select_model(self):
        """Автоматически выбирает первую vision-модель."""
        try:
            models = self.client.models.list()
            vision_keywords = ['vision', 'llava', 'bakllava', 'minicpm', 'qwen-vl', 'internvl']

            # Ищем vision модель
            for m in models.data:
                name_lower = m.id.lower()
                if any(kw in name_lower for kw in vision_keywords):
                    self.model = m.id
                    if self.verbose:
                        print(f"Выбрана vision модель: {self.model}")
                    return

            # Если vision не нашли — берём первую
            if models.data:
                self.model = models.data[0].id
                if self.verbose:
                    print(f"Vision модель не найдена, используем: {self.model}")
        except Exception as e:
            if self.verbose:
                print(f"Не удалось получить список моделей: {e}")
            self.model = "default"

    def verify_gate_pass(self, frame: np.ndarray) -> Optional[bool]:
        """
        Спрашивает vision-модель: это пролёт через ворота?

        Returns:
            True/False или None если не удалось получить ответ
        """
        if self.model is None:
            return None

        try:
            img_b64 = frame_to_base64(frame)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "This is a frame from an FPV drone racing video. "
                                    "The drone is flying through or approaching LED gate markers. "
                                    "Is the drone currently passing through or about to pass through a gate? "
                                    "Answer with just YES or NO."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=10,
                temperature=0,
            )

            answer = response.choices[0].message.content.strip().upper()
            if self.verbose:
                print(f"  LM Studio ответ: '{answer}'")
            return "YES" in answer

        except Exception as e:
            if self.verbose:
                print(f"  LM Studio ошибка: {e}")
            return None

    def is_available(self) -> bool:
        """Проверяет доступность LM Studio."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
