from pathlib import Path
from typing import Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LlavaForConditionalGeneration


class ManeuverDecider:
    MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    MANEUVER_PROMPT = (
        "You are the driver of the vehicle and you see this from your perspective. "
        "Describe the scene and choose between these driving maneuvers: go, stop. "
    )

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")
        self.prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.MANEUVER_PROMPT},
                    {"type": "image"},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

    def decide(
        self,
        image: Union[Image.Image, str, Path],
    ) -> Tuple[str, Image.Image]:
        raw_image = self._load_image(image)
        inputs = self.processor(
            images=raw_image,
            text=self.prompt,
            return_tensors="pt",
        ).to("cuda", torch.float16)

        output = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        decision = decoded.split("ASSISTANT:")[-1].strip()
        return decision, raw_image

    @staticmethod
    def _load_image(image: Union[Image.Image, str, Path]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    @staticmethod
    def save_visualization(
        decision: str,
        image: Image.Image,
        output_file: Union[str, Path] = "llava_decision.png",
    ) -> None:
        resized_image = image.copy()
        resized_image.thumbnail((1000, 700))

        font = ImageFont.load_default()
        draw = ImageDraw.Draw(resized_image)
        label = f"LLaVA decision: {decision}"
        wrapped_label = ManeuverDecider._wrap_text(
            label,
            draw,
            font,
            max(1, resized_image.width - 60),
        )
        text_bounds = draw.multiline_textbbox(
            (0, 0),
            wrapped_label,
            font=font,
            spacing=4,
        )
        text_height = max(150, text_bounds[3] - text_bounds[1] + 40)
        canvas = Image.new(
            "RGB",
            (resized_image.width, resized_image.height + text_height),
            "white",
        )
        canvas.paste(resized_image, (0, 0))

        draw = ImageDraw.Draw(canvas)
        draw.multiline_text(
            (30, resized_image.height + 20),
            wrapped_label,
            fill="black",
            font=font,
            spacing=4,
        )
        canvas.save(output_file)

    @staticmethod
    def _wrap_text(
        text: str,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        max_width: int,
    ) -> str:
        lines = []
        current_line = ""
        for word in text.split():
            candidate = f"{current_line} {word}".strip()
            if draw.textlength(candidate, font=font) <= max_width:
                current_line = candidate
                continue

            if current_line:
                lines.append(current_line)
                current_line = ""

            for character in word:
                candidate = current_line + character
                if current_line and draw.textlength(candidate, font=font) > max_width:
                    lines.append(current_line)
                    current_line = character
                else:
                    current_line = candidate

        if current_line:
            lines.append(current_line)
        return "\n".join(lines)


def main() -> None:
    decider = ManeuverDecider()
    print("LLaVA loaded")

    image_file = "examples/scene1.png"
    decision, raw_image = decider.decide(image_file)
    print("Decision:", decision)

    output_file = "results/scene1_decision.png"
    decider.save_visualization(decision, raw_image, output_file)
    print(f"Saved visualization to: {output_file}")


if __name__ == "__main__":
    main()