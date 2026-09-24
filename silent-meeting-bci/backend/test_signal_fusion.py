import numpy as np
import pytest

from backend.signal_fusion import (
    MultiModalFusion,
    SignalFusion,
    SignalSource,
    TextSegment,
    WeakSignalEnhancer,
)


def make_segment(text, confidence=0.9, timestamp=100.0,
                 source=SignalSource.BCI, **kwargs):
    return TextSegment(
        text=text,
        confidence=confidence,
        timestamp=timestamp,
        source=source,
        **kwargs,
    )


class TestSignalFusion:
    def test_matched_pair_weighted_confidence(self):
        fusion = SignalFusion(bci_weight=0.4, audio_weight=0.6)
        fusion.add_bci_segment(make_segment("hello", 0.8, 100.0, SignalSource.BCI))
        fusion.add_audio_segment(make_segment("hello", 0.9, 100.2, SignalSource.AUDIO))
        result = fusion.fuse()
        assert len(result.segments) == 1
        seg = result.segments[0]
        assert seg.source == SignalSource.FUSED
        assert seg.confidence == pytest.approx(0.8 * 0.4 + 0.9 * 0.6)
        assert result.overall_confidence == pytest.approx(seg.confidence)

    def test_single_source_confidence_not_scaled_by_weight(self):
        fusion = SignalFusion()
        fusion.add_audio_segment(make_segment("only audio", 0.9, 100.0, SignalSource.AUDIO))
        result = fusion.fuse()
        assert len(result.segments) == 1
        assert result.segments[0].confidence == pytest.approx(0.9)
        assert result.segments[0].text == "only audio"

        fusion2 = SignalFusion()
        fusion2.add_bci_segment(make_segment("only bci", 0.7, 100.0, SignalSource.BCI))
        result2 = fusion2.fuse()
        assert result2.segments[0].confidence == pytest.approx(0.7)

    def test_buffers_consumed_after_fuse(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(make_segment("hello", 0.8, 100.0, SignalSource.BCI))
        first = fusion.fuse()
        assert len(first.segments) == 1
        second = fusion.fuse()
        assert second.segments == []
        assert second.fused_text == ""
        assert second.overall_confidence == 0.0
        assert len(fusion.get_fused_segments()) == 1

    def test_relative_timestamps_not_discarded(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(make_segment("a", 0.8, 0.0, SignalSource.BCI))
        fusion.add_audio_segment(make_segment("a", 0.9, 0.5, SignalSource.AUDIO))
        result = fusion.fuse()
        assert len(result.segments) == 1
        assert result.segments[0].source == SignalSource.FUSED

    def test_alignment_respects_max_delay_inclusively(self):
        fusion = SignalFusion(max_delay=1.0)
        fusion.add_bci_segment(make_segment("x", 0.8, 100.0, SignalSource.BCI))
        fusion.add_audio_segment(make_segment("x", 0.9, 101.0, SignalSource.AUDIO))
        result = fusion.fuse()
        assert len(result.segments) == 1
        assert result.segments[0].source == SignalSource.FUSED

    def test_unmatched_segments_paired_with_none(self):
        fusion = SignalFusion(max_delay=0.5)
        fusion.add_bci_segment(make_segment("bci only", 0.8, 100.0, SignalSource.BCI))
        fusion.add_audio_segment(make_segment("audio only", 0.9, 105.0, SignalSource.AUDIO))
        result = fusion.fuse()
        assert len(result.segments) == 2
        texts = {seg.text for seg in result.segments}
        assert texts == {"bci only", "audio only"}

    def test_zero_start_time_not_treated_as_missing(self):
        fusion = SignalFusion()
        bci = make_segment("hi", 0.8, 0.0, SignalSource.BCI,
                           start_time=0.0, end_time=0.5)
        audio = make_segment("hi", 0.9, 0.1, SignalSource.AUDIO,
                             start_time=0.1, end_time=0.6)
        fusion.add_bci_segment(bci)
        fusion.add_audio_segment(audio)
        result = fusion.fuse()
        seg = result.segments[0]
        assert seg.start_time == pytest.approx(0.0)
        assert seg.end_time == pytest.approx(0.6)

    def test_merge_texts_similar_and_different(self):
        fusion = SignalFusion()
        assert fusion._merge_texts("", "") == ""
        assert fusion._merge_texts("hello", "") == "hello"
        assert fusion._merge_texts("", "world") == "world"
        similar = fusion._merge_texts("the quick fox", "the quick brown fox")
        assert similar == "the quick brown fox"
        different = fusion._merge_texts("alpha beta", "gamma delta")
        assert different == "alpha beta gamma delta"

    def test_clear_buffers(self):
        fusion = SignalFusion()
        fusion.add_bci_segment(make_segment("a", 0.8, 1.0, SignalSource.BCI))
        fusion.fuse()
        fusion.clear_buffers()
        assert fusion.get_fused_segments() == []


class TestWeakSignalEnhancer:
    def test_weak_signal_amplified(self):
        enhancer = WeakSignalEnhancer(noise_floor=0.1, enhancement_factor=2.0)
        signal = np.full(100, 0.05)
        enhanced, _ = enhancer.enhance(signal, 0.8)
        assert np.mean(np.abs(enhanced)) > np.mean(np.abs(signal))

    def test_enhancement_does_not_boost_confidence(self):
        enhancer = WeakSignalEnhancer(noise_floor=0.1, enhancement_factor=2.0)
        signal = np.full(100, 0.05)
        _, confidence = enhancer.enhance(signal, 0.6)
        assert 0.0 <= confidence <= 0.6

    def test_strong_signal_untouched(self):
        enhancer = WeakSignalEnhancer(noise_floor=0.01)
        signal = np.full(100, 0.5)
        enhanced, confidence = enhancer.enhance(signal, 0.7)
        np.testing.assert_array_equal(enhanced, signal)
        assert confidence == pytest.approx(0.7)

    def test_signal_power_is_mean_square(self):
        enhancer = WeakSignalEnhancer(noise_floor=0.0)
        signal = np.array([1.0, -1.0, 2.0, -2.0])
        enhancer.enhance(signal, 0.9)
        assert enhancer.get_average_signal_power() == pytest.approx(2.5)

    def test_average_signal_power_empty(self):
        enhancer = WeakSignalEnhancer()
        assert enhancer.get_average_signal_power() == 0.0


class TestMultiModalFusion:
    def test_weights_normalized(self):
        fusion = MultiModalFusion()
        fusion.add_modality("bci", [make_segment("a", 0.8, 1.0)], weight=1.0)
        fusion.add_modality("audio", [make_segment("b", 0.6, 5.0)], weight=1.0)
        result = fusion.fuse_all()
        confidences = sorted(seg.confidence for seg in result.segments)
        assert confidences[0] == pytest.approx(0.3)
        assert confidences[1] == pytest.approx(0.4)
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_single_modality_confidence_preserved(self):
        fusion = MultiModalFusion()
        fusion.add_modality("audio", [make_segment("hello", 0.9, 1.0)], weight=2.0)
        result = fusion.fuse_all()
        assert result.segments[0].confidence == pytest.approx(0.9)

    def test_fused_text_and_sorting(self):
        fusion = MultiModalFusion()
        fusion.add_modality("m1", [make_segment("later", 0.8, 10.0)])
        fusion.add_modality("m2", [make_segment("earlier", 0.8, 5.0)])
        result = fusion.fuse_all()
        assert result.fused_text == "earlier later"

    def test_merge_overlapping_same_speaker(self):
        fusion = MultiModalFusion()
        seg1 = make_segment("hello", 0.8, 10.0, speaker_id="s1",
                            start_time=9.0, end_time=10.5)
        seg2 = make_segment("world", 0.6, 10.5, speaker_id="s1",
                            start_time=10.2, end_time=11.0)
        fusion.add_modality("m", [seg1, seg2])
        result = fusion.fuse_all()
        assert len(result.segments) == 1
        merged = result.segments[0]
        assert merged.text == "hello world"
        assert merged.confidence == pytest.approx(0.7)
        assert merged.end_time == pytest.approx(11.0)

    def test_no_merge_across_speakers(self):
        fusion = MultiModalFusion()
        fusion.add_modality("m", [
            make_segment("a", 0.8, 10.0, speaker_id="s1"),
            make_segment("b", 0.8, 10.2, speaker_id="s2"),
        ])
        result = fusion.fuse_all()
        assert len(result.segments) == 2

    def test_empty_fusion(self):
        fusion = MultiModalFusion()
        result = fusion.fuse_all()
        assert result.segments == []
        assert result.fused_text == ""
        assert result.overall_confidence == 0.0
