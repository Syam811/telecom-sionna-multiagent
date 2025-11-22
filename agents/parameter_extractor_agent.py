import re
import logging

class ParameterExtractorAgent:
    def __init__(self):
        self.logger = logging.getLogger("ParameterExtractorAgent")

    def run(self, task):
        prompt = task.prompt.lower()
        params = {}

        # -------------------------
        # CONSTELLATION
        # -------------------------
        if task.task_type == "constellation":
            params["modulation"] = self._extract_modulation(prompt)
            params["snr_db"] = self._extract_snr(prompt, default=10)
            self.logger.info(f"Extracted params: {params}")
            task.parameters = params
            return task

        # -------------------------
        # BER
        # -------------------------
        if task.task_type == "ber":
            params["modulation"] = self._extract_modulation(prompt)
            params["channel"] = self._extract_channel(prompt)
            params["snr_db_list"] = self._extract_snr_list(prompt)
            self.logger.info(f"Extracted params: {params}")
            task.parameters = params
            return task

        # -------------------------
        # MIMO COMPARISON
        # -------------------------
        if task.task_type == "mimo_comparison":
            params["modulation"] = self._extract_modulation(prompt)
            params["snr_db_list"] = self._extract_snr_list(prompt)
            params["configs"] = self._extract_mimo_configs(prompt)
            self.logger.info(f"Extracted params: {params}")
            task.parameters = params
            return task

        # -------------------------
        # SINGLE RADIO MAP
        # -------------------------
        if task.task_type == "radiomap":
            params["tx_pos"] = self._extract_single_tx(prompt)
            self.logger.info(f"Extracted params: {params}")
            task.parameters = params
            return task

        # -------------------------
        # MULTI RADIO MAP
        # -------------------------
        if task.task_type == "multi_radio_map":
            params["tx_positions"] = self._extract_multi_tx(prompt)
            params["combine_mode"] = self._extract_combine_mode(prompt)
            self.logger.info(f"Extracted params: {params}")
            task.parameters = params
            return task

        # fallback
        task.parameters = {}
        self.logger.info("Extracted params: {}")
        return task


    # -------------------------
    # HELPER FUNCTIONS
    # -------------------------

    def _extract_modulation(self, text):
        for m in ["qpsk", "16qam", "64qam", "256qam"]:
            if m in text:
                return m
        return "qpsk"

    def _extract_snr(self, text, default=10):
        match = re.search(r"snr\s*(-?\d+)", text)
        if match:
            return float(match.group(1))
        return default

    def _extract_snr_list(self, text):
        nums = re.findall(r"-?\d+", text)
        nums = [int(n) for n in nums]

        if len(nums) >= 2:
            return nums
        return [-5, 0, 5, 10, 15]

    def _extract_channel(self, text):
        if "rayleigh" in text:
            return "rayleigh"
        return "awgn"

    def _extract_mimo_configs(self, text):
        configs = []
        matches = re.findall(r"(\d+)x(\d+)", text)
        for nt, nr in matches:
            configs.append({"nt": int(nt), "nr": int(nr)})
        if configs:
            return configs
        return [{"nt": 1, "nr": 1}, {"nt": 4, "nr": 4}]

    def _extract_single_tx(self, text):
        match = re.findall(r"\(([-\d]+),\s*([-\d]+),\s*([-\d]+)\)", text)
        if match:
            x, y, z = match[0]
            return [float(x), float(y), float(z)]
        return [0, 0, 10]

    def _extract_multi_tx(self, text):
        matches = re.findall(r"\(([-\d]+),\s*([-\d]+),\s*([-\d]+)\)", text)
        if matches:
            return [[float(a), float(b), float(c)] for (a, b, c) in matches]
        return [[0, 0, 10], [60, 0, 10], [-60, 0, 10]]

    def _extract_combine_mode(self, text):
        if "sum" in text or "add" in text:
            return "sum"
        return "max"
