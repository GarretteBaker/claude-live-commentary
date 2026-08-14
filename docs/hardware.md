# Hardware Recommendation: Multi-Channel Wired Transcription Rig (~$10k budget)

## 1. TL;DR

Skip the Meeting Owl and the ceiling-array ecosystem entirely — both collapse to a single processed mono mix by the time they reach a Linux USB port, which destroys the one thing your software needs: per-channel attribution. Buy **one Focusrite Scarlett 18i20 4th Gen** ($749.99, Sweetwater) — the best-supported multichannel interface on Linux, with a dedicated mainline kernel driver and `alsa-scarlett-gui` — optionally expanded to 16 sample-synchronous mic channels via a **Scarlett OctoPre** over ADAT ($579.99). Populate it with **cardioid boundary mics (Shure MX393/C, $316 ea.)** — one per 2–3 seated people, cardioid rather than omni specifically to sharpen loudest-channel attribution — plus SM58s for the lecturer and audience passing. The honest recommended build lands at **~$6,400**; there is no useful way to spend the remaining $3,600 on this problem without breaking the per-channel constraint, so I recommend banking it (or buying spares/a second room kit). A lean build at ~$4,300 keeps the same interface with 8 channels; a bargain build at ~$1,500 exists if needed.

## 2. The Owl Verdict

**No.** The Meeting Owl 3 ($999–1,099, [Owl Labs](https://owllabs.com/products/meeting-owl-3)/[Best Buy](https://www.bestbuy.com/product/owl-labs-meeting-owl-3-gray/J3R8T84726)) has an 8-mic array, but all beamforming, echo cancellation, "voice equalization," and noise reduction happen onboard, and it presents to the computer as a standard USB speakerphone — a **single processed mix channel**, never the 8 raw mics$_{92\%}$ (Owl publishes no multichannel USB spec anywhere, and every deployment reference treats it as a mono speakerphone). Its DSP is tuned for far-end human intelligibility on calls, not ASR — AGC and aggressive noise reduction typically *hurt* raw transcription accuracy, and per-channel speaker attribution is definitionally impossible on one channel. It works fine as a class-compliant webcam+speakerphone on Linux, but for this system it's a $1,099 way to lose your diarization signal.

The fancy arrays fail the same test for a different reason: the **Shure MXA920** ($4,732, [SoundPro](https://soundpro.com/products/mxa920w-r-usb-v-bundle)/[Full Compass](https://www.fullcompass.com/prod/627286-shure-mxa920-r-usb-v-mxa920-r-array-microphone-and-aniusb-matrix-interface)) genuinely exposes 8 steerable lobes as separate channels — **but only over Dante**. The USB bridge everyone bundles it with, the ANIUSB-MATRIX ($845, [B&H](https://www.bhphotovideo.com/c/product/1365410-REG/shure_aniusb_matrix_dante_audio_network_interface.html)), is a 1-in/1-out USB device: it automixes the lobes into a single channel. There is no Dante Virtual Soundcard for Linux, Audinate's AVIO USB adapters are 2×2, and multichannel Dante→USB hardware bridges start around $1–2k more. So MXA920 per-lobe capture on a Fedora laptop is ~$7k+ and still experimental (the open-source *Inferno* Dante implementation exists but is not something to build a live system on). Nureva HDL300 ($3,599 MSRP, [B&H](https://www.bhphotovideo.com/c/product/1638724-REG/nureva_hdl300_conferencing_soundbar.html)) and MXA710 are single-mix-over-USB for the same bridging reason. Wired discrete mics win decisively here.

## 3. Packages

### Package A — Recommended, ~$6,400 (16 sample-locked channels)

| Item | Role | Price | Linux notes |
|---|---|---|---|
| Focusrite Scarlett 18i20 4th Gen | 8 mic pres, single USB device, single clock | $749.99 ([Sweetwater](https://www.sweetwater.com/store/detail/Scar18i20G4--focusrite-scarlett-18i20-fourth-generation-usb-audio-interface)) | Class-compliant; **best-in-class Linux support** — Geoffrey Bennett's kernel driver + [alsa-scarlett-gui](https://github.com/geoffreybennett/alsa-scarlett-gui) covers the big Gen-4 units ([linux-fcp](https://github.com/geoffreybennett/linux-fcp)); Fedora 44's kernel is new enough. PipeWire sees all inputs as one multichannel source. |
| Focusrite Scarlett OctoPre | +8 preamps via ADAT lightpipe → 16 mic channels | $579.99 ([Sweetwater](https://www.sweetwater.com/store/detail/ScarOctoPre--focusrite-scarlett-octopre-mic-preamp)) | Zero driver risk — it's analog→ADAT, invisible to the OS; clocked from the 18i20, so all 16 channels stay sample-aligned. |
| 12× Shure MX393/**C** (cardioid) boundary | Table mics, 1 per 2–3 people | $3,792 @ $316 ([Sweetwater](https://www.sweetwater.com/store/detail/MX393O--shure-mx393-o-microflex-omnidirectional-boundary-microphone)/[Adorama](https://www.adorama.com/shmx393o.html)) | Passive XLR, no Linux risk. Needs +48V phantom (both Focusrite boxes supply it). **Cardioid, not omni** — you want directivity for loudest-channel attribution. |
| 1× Shure SM58 + On-Stage MS7701B boom | Lecturer close-mic (strongest attribution channel) | $109 + $31.95 ([Sweetwater](https://www.sweetwater.com/store/detail/SM58--shure-sm58-cardioid-dynamic-vocal-microphone)) | Dynamic, phantom-safe, indestructible. |
| 2× Shure SM58 + 1 spare stand | Audience Q&A passing mics | $218 + $32 (Sweetwater) | Same. |
| 2× Seismic Audio SARLX-8x50 (8-ch XLR snake, 50 ft) | Clean runs from interface desk down the seminar table | $239.98 ([Seismic Audio](https://www.seismicaudiospeakers.com/products/8-channel-xlr-snake-cable-50-feet)) | Passive copper. |
| 12× 25 ft + 4× 10 ft XLR (Hosa/Monoprice) | Snake fantail → mic positions | ~$250 | — |
| Gaff tape, mic clips, spare TRS/USB-C cables | Misc | ~$100 | — |
| **Total** | | **≈ $6,410** | |

15 attributed channels (12 boundary + lecturer + 2 passing), one channel spare. **The remaining ~$3,600: don't spend it.** The marginal dollar past this point buys either redundancy (fine: +$1,300 for a spare 18i20+SM58s) or Dante gear that Linux can't ingest per-channel.

### Package B — Lean, ~$4,300 (8 channels, same interface quality)

| Item | Role | Price | Linux notes |
|---|---|---|---|
| Focusrite Scarlett 18i20 4th Gen | 8 mic pres | $749.99 (Sweetwater) | As above — don't cheap out on the one item where Linux support varies. |
| 8× Shure MX393/C boundary | Table coverage, 1 per 3–4 people | $2,528 | As above. |
| 3× Shure SM58 + 2 stands | Lecturer + 2 passing | $391 (Sweetwater) | — |
| 1× SARLX-8x50 snake + 8 XLR cables | Cabling | ~$290 | — |
| Misc | | ~$80 | — |
| **Total** | | **≈ $4,040** | |

### Package C — Bargain floor, ~$1,500 (if the money is better spent elsewhere)

| Item | Role | Price | Linux notes |
|---|---|---|---|
| Behringer UMC1820 | 8 Midas-designed pres | ~$230 ([Sweetwater](https://www.sweetwater.com/store/detail/UMC1820--behringer-u-phoria-umc1820-usb-audio-interface)) | Class-compliant; multiple confirmed working reports on [Linux Mint/Ardour](https://forums.linuxmint.com/viewtopic.php?t=300904), all knobs hardware so no control software needed; a few forum reports of input-mapping quirks — more variance than Focusrite. |
| 8× Samson CM11B boundary | Table mics (omni — weaker attribution) | $799.92 ([B&H](https://www.bhphotovideo.com/c/product/330988-REG/Samson_SACM11B_CM11B_Omnidirectional_Boundary_Microphone.html)) | Passive XLR, phantom required. Omni-only at this tier hurts channel separation. |
| 3× Behringer XM8500 + 2 stands | Lecturer + passing | ~$124 ([Sweetwater](https://www.sweetwater.com/store/detail/XM8500--behringer-xm8500-handheld-dynamic-vocal-microphone)) | SM58 clone, $19.90 each, genuinely fine for ASR. |
| Snake + cables | | ~$290 | — |
| **Total** | | **≈ $1,445** | |

Middle-tier interface alternative: **Audient EVO 16** ($584–675, [Sweetwater](https://www.sweetwater.com/store/detail/EVO16--audient-evo-16-usb-audio-interface)) is class-compliant and works, but its routing/mixer software is Win/Mac-only with no Linux control panel — the 18i20 dominates it at similar cost. **MOTU UltraLite mk5** is disqualified (only 2 mic preamps; its class-compliant mode is also Apple-oriented per [MOTUnation](https://www.motunation.com/forum/viewtopic.php?t=69019)). RME is superbly Linux-friendly in class-compliant mode but you'd pay ~$700/preamp-pair; wrong shape for this problem.

## 4. Key Risks & Unknowns

- **Owl mono-mix claim**: confident$_{92\%}$ but Owl Labs publishes no USB descriptor spec; a 10-minute `pactl list sources` test on any borrowed Owl settles it before anyone spends $1,099.
- **18i20 4th Gen kernel support**: basic 18-channel capture is class-compliant out of the box; the *full* mixer control path needs the FCP driver stack — on Fedora 44 (kernel 7.1) this is present, but verify `alsa-scarlett-gui` sees the unit on day one. Sources: [Linux Audio Wiki](https://wiki.linuxaudio.org/hw/focusrite_scarlett), [linuxmusicians thread](https://linuxmusicians.com/viewtopic.php?t=27505).
- **Attribution physics**: loudest-channel attribution degrades when adjacent boundary mics are <~1.5 m apart or when the room is very live. Cardioid boundaries + deliberate spacing (one mic per table "zone," pointed away from neighbors) is the mitigation; budget one pilot session to tune spacing before finalizing mic count.
- **Clocking**: do *not* run the two existing Volt 2s alongside the 18i20 for attributed channels — three free-running USB clocks drift relative to each other, and word-level loudest-channel comparison assumes sample alignment. Keep the Volts as spares. The OctoPre avoids this entirely (ADAT slaves to the 18i20's clock).
- **Phantom power**: every boundary condenser listed needs +48V; both Focusrite boxes and the UMC1820 provide it on all mic inputs. Dynamics (SM58/XM8500) on the same phantom bank are safe.
- **Prices**: all quoted 2026-08 street prices; MX393 units fluctuate ±$20 across Sweetwater/Adorama/B&H, and Behringer stock is erratic.

Sources: [Sweetwater 18i20](https://www.sweetwater.com/store/detail/Scar18i20G4--focusrite-scarlett-18i20-fourth-generation-usb-audio-interface) · [linux-fcp](https://github.com/geoffreybennett/linux-fcp) · [alsa-scarlett-gui](https://github.com/geoffreybennett/alsa-scarlett-gui) · [SoundPro MXA920+ANIUSB bundle](https://soundpro.com/products/mxa920w-r-usb-v-bundle) · [B&H ANIUSB-MATRIX](https://www.bhphotovideo.com/c/product/1365410-REG/shure_aniusb_matrix_dante_audio_network_interface.html) · [Owl Labs Meeting Owl 3](https://owllabs.com/products/meeting-owl-3) · [Sweetwater MX393](https://www.sweetwater.com/store/detail/MX393O--shure-mx393-o-microflex-omnidirectional-boundary-microphone) · [A-T ES945 press pricing](https://www.audio-technica.com/en-us/press/audio-technica-unveils-its-next-generation-es945-and-es947-boundary-microphone-variations) · [Sweetwater UMC1820](https://www.sweetwater.com/store/detail/UMC1820--behringer-u-phoria-umc1820-usb-audio-interface) · [Linux Mint UMC1820 report](https://forums.linuxmint.com/viewtopic.php?t=300904) · [Sweetwater EVO 16](https://www.sweetwater.com/store/detail/EVO16--audient-evo-16-usb-audio-interface) · [Sweetwater OctoPre](https://www.sweetwater.com/store/detail/ScarOctoPre--focusrite-scarlett-octopre-mic-preamp) · [Seismic Audio snake](https://www.seismicaudiospeakers.com/products/8-channel-xlr-snake-cable-50-feet) · [B&H Nureva HDL300](https://www.bhphotovideo.com/c/product/1638724-REG/nureva_hdl300_conferencing_soundbar.html) · [Sweetwater SM58](https://www.sweetwater.com/store/detail/SM58--shure-sm58-cardioid-dynamic-vocal-microphone) · [Sweetwater XM8500](https://www.sweetwater.com/store/detail/XM8500--behringer-xm8500-handheld-dynamic-vocal-microphone) · [Sweetwater MS7701B](https://www.sweetwater.com/store/detail/MicStdFBoomL--on-stage-stands-ms7701b-tripod-microphone-stand-black)
