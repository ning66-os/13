import unittest

import numpy as np

from backend.signal_fusion import (
    MultiModalFusion,
    SignalFusion,
    SignalSource,
    TextSegment,
    WeakSignalEnhancer,
)


def make_segment(text, confidence, timestamp, source, speaker_id=None,
                 start_time=None, end_time=None):
    return TextSegment(
        text=text,
        confidence=confidence,
        timestamp=timestamp,
        source=source,
        speaker_id=speaker_id,
        start_time=start_time,
        end_time=end_time,
    )


class SignalFusionTest(unittest.TestCase):
    def setUp(self):
        self.fusion = SignalFusion(bci_weight=0.4, audio_weight=0.6,
                                   time_window=2.0, max_delay=1.0)

    def test_paired_segments_fuse_with_normalized_confidence(self):
        self.fusion.add_bci_segment(
            make_segment("hello world", 0.8, 100.0, SignalSource.BCI))
        self.fusion.add_audio_segment(
            make_segment("hello world", 0.9, 100.2, SignalSource.AUDIO))

        result = self.fusion.fuse()

        self.assertEqual(len(result.segments), 1)
        seg = result.segments[0]
        self.assertEqual(seg.source, SignalSource.FUSED)
        self.assertEqual(seg.text, "hello world")
        expected = (0.8 * 0.4 + 0.9 * 0.6) / 1.0
        self.assertAlmostEqual(seg.confidence, expected, places=6)
        self.assertAlmostEqual(result.overall_confidence, expected, places=6)
        self.assertIsInstance(result.overall_confidence, float)

    def test_unnormalized_weights_still_bounded(self):
        fusion = SignalFusion(bci_weight=2.0, audio_weight=3.0)
        fusion.add_bci_segment(make_segment("a", 0.9, 10.0, SignalSource.BCI))
        fusion.add_audio_segment(make_segment("a", 0.8, 10.1, SignalSource.AUDIO))

        result = fusion.fuse()

        expected = (0.9 * 2.0 + 0.8 * 3.0) / 5.0
        self.assertAlmostEqual(result.segments[0].confidence, expected, places=6)
        self.assertLessEqual(result.segments[0].confidence, 1.0)

    def test_single_source_confidence_not_penalized(self):
        self.fusion.add_audio_segment(
            make_segment("only audio", 0.9, 50.0, SignalSource.AUDIO))
        self.fusion.add_bci_segment(
            make_segment("only bci", 0.7, 60.0, SignalSource.BCI))

        result = self.fusion.fuse()

        confidences = {seg.text: seg.confidence for seg in result.segments}
        self.assertAlmostEqual(confidences["only audio"], 0.9, places=6)
        self.assertAlmostEqual(confidences["only bci"], 0.7, places=6)

    def test_fused_segments_keep_chronological_order(self):
        self.fusion.add_audio_segment(
            make_segment("late audio", 0.9, 105.0, SignalSource.AUDIO))
        self.fusion.add_bci_segment(
            make_segment("early", 0.8, 100.0, SignalSource.BCI))
        self.fusion.add_audio_segment(
            make_segment("early", 0.9, 100.1, SignalSource.AUDIO))

        result = self.fusion.fuse()

        timestamps = [seg.timestamp for seg in result.segments]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(result.fused_text, "early late audio")

    def test_buffers_consumed_after_fuse(self):
        self.fusion.add_bci_segment(
            make_segment("once", 0.8, 100.0, SignalSource.BCI))

        first = self.fusion.fuse()
        second = self.fusion.fuse()

        self.assertEqual(len(first.segments), 1)
        self.assertEqual(len(second.segments), 0)
        self.assertEqual(second.fused_text, "")
        self.assertEqual(len(self.fusion.get_fused_segments()), 1)

    def test_clean_old_segments_uses_relative_reference(self):
        self.fusion.add_bci_segment(
            make_segment("ancient", 0.8, 10.0, SignalSource.BCI))
        self.fusion.add_bci_segment(
            make_segment("recent", 0.8, 100.0, SignalSource.BCI))

        result = self.fusion.fuse()

        self.assertEqual([seg.text for seg in result.segments], ["recent"])

    def test_relative_timestamps_not_wiped_by_wall_clock(self):
        # 小的相对时间戳不应被当作“过期”而清空
        self.fusion.add_audio_segment(
            make_segment("relative", 0.9, 1.5, SignalSource.AUDIO))

        result = self.fusion.fuse()

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.fused_text, "relative")

    def test_pairing_respects_max_delay(self):
        self.fusion.add_bci_segment(
            make_segment("bci text", 0.8, 100.0, SignalSource.BCI))
        self.fusion.add_audio_segment(
            make_segment("audio text", 0.9, 105.0, SignalSource.AUDIO))

        result = self.fusion.fuse()

        self.assertEqual(len(result.segments), 2)

    def test_fused_pair_time_bounds(self):
        self.fusion.add_bci_segment(make_segment(
            "x", 0.8, 100.0, SignalSource.BCI, start_time=99.5, end_time=100.5))
        self.fusion.add_audio_segment(make_segment(
            "x", 0.9, 100.2, SignalSource.AUDIO, start_time=99.8, end_time=101.0))

        seg = self.fusion.fuse().segments[0]

        self.assertAlmostEqual(seg.start_time, 99.5)
        self.assertAlmostEqual(seg.end_time, 101.0)
        self.assertAlmostEqual(seg.timestamp, 100.0)

    def test_merge_texts_prefers_longer_when_similar(self):
        merged = self.fusion._merge_texts("the cat", "the cat sat down")
        self.assertEqual(merged, "the cat sat down")

    def test_merge_texts_concatenates_when_different(self):
        merged = self.fusion._merge_texts("alpha beta", "gamma delta")
        self.assertEqual(merged, "alpha beta gamma delta")

    def test_merge_texts_handles_empty(self):
        self.assertEqual(self.fusion._merge_texts("", ""), "")
        self.assertEqual(self.fusion._merge_texts("a", ""), "a")
        self.assertEqual(self.fusion._merge_texts("", "b"), "b")

    def test_clear_buffers(self):
        self.fusion.add_bci_segment(
            make_segment("x", 0.5, 1.0, SignalSource.BCI))
        self.fusion.fuse()
        self.fusion.clear_buffers()
        self.assertEqual(self.fusion.get_fused_segments(), [])


class WeakSignalEnhancerTest(unittest.TestCase):
    def test_weak_signal_is_amplified(self):
        enhancer = WeakSignalEnhancer(noise_floor=0.1, enhancement_factor=2.0)
        signal = np.full(16, 0.2)  # 功率 0.04 < 0.1，但平均幅值 0.2 > 0.1

        enhanced, confidence = enhancer.enhance(signal, 0.5)

        self.assertGreater(np.mean(np.abs(enhanced)), np.mean(np.abs(signal)))
        self.assertLessEqual(confidence, 1.0)

    def test_strong_signal_passes_through(self):
        enhancer = WeakSignalEnhancer(noise_floor=0.1)
        signal = np.full(16, 0.9)  # 功率 0.81 > 0.1

        enhanced, confidence = enhancer.enhance(signal, 0.7)

        np.testing.assert_array_equal(enhanced, signal)
        self.assertEqual(confidence, 0.7)

    def test_signal_power_uses_mean_square(self):
        enhancer = WeakSignalEnhancer()
        enhancer.enhance(np.array([0.5, -0.5, 0.5, -0.5]), 0.5)
        self.assertAlmostEqual(enhancer.get_average_signal_power(), 0.25)

    def test_history_is_bounded(self):
        enhancer = WeakSignalEnhancer()
        for _ in range(150):
            enhancer.enhance(np.ones(4), 0.5)
        self.assertEqual(len(enhancer.signal_history), 100)

    def test_average_power_empty_history(self):
        self.assertEqual(WeakSignalEnhancer().get_average_signal_power(), 0.0)


class MultiModalFusionTest(unittest.TestCase):
    def test_fuse_all_sorts_and_computes_confidence(self):
        fusion = MultiModalFusion()
        fusion.add_modality("bci", [
            make_segment("second", 0.8, 2.0, SignalSource.BCI, speaker_id="a"),
            make_segment("first", 0.6, 1.0, SignalSource.BCI, speaker_id="b"),
        ])
        fusion.add_modality("audio", [
            make_segment("third", 1.0, 10.0, SignalSource.AUDIO, speaker_id="a"),
        ], weight=0.5)

        result = fusion.fuse_all()

        timestamps = [seg.timestamp for seg in result.segments]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertLessEqual(result.overall_confidence, 1.0)
        self.assertIsInstance(result.overall_confidence, float)

    def test_overlapping_same_speaker_merges_without_duplication(self):
        fusion = MultiModalFusion()
        fusion.add_modality("audio", [
            make_segment("hello", 0.9, 1.0, SignalSource.AUDIO, speaker_id="s1"),
            make_segment("hello", 0.8, 1.5, SignalSource.AUDIO, speaker_id="s1"),
        ])
        fusion.add_modality("bci", [
            make_segment("world", 0.7, 1.7, SignalSource.BCI, speaker_id="s1"),
        ])

        result = fusion.fuse_all()

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "hello world")
        self.assertEqual(result.segments[0].source, SignalSource.FUSED)

    def test_different_speakers_not_merged(self):
        fusion = MultiModalFusion()
        fusion.add_modality("audio", [
            make_segment("hi", 0.9, 1.0, SignalSource.AUDIO, speaker_id="s1"),
            make_segment("yo", 0.9, 1.2, SignalSource.AUDIO, speaker_id="s2"),
        ])

        result = fusion.fuse_all()

        self.assertEqual(len(result.segments), 2)

    def test_empty_fusion(self):
        result = MultiModalFusion().fuse_all()
        self.assertEqual(result.segments, [])
        self.assertEqual(result.fused_text, "")
        self.assertEqual(result.overall_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
