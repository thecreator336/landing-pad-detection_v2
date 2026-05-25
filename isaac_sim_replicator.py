import omni.replicator.core as rep
import omni.usd
import time
import os

# ─── ONLY CHANGE THIS EACH SESSION ───────────────────────────────────────────
BATCH_INDEX = 0        # 0-3 yellow, 4-6 white, 7-9 dirty
# ─────────────────────────────────────────────────────────────────────────────

FRAMES_PER_BATCH = 1000
OUTPUT_ROOT      = "C:/landing_dataset_2"

batch_dir = os.path.join(OUTPUT_ROOT, f"batch_{BATCH_INDEX:04d}")
os.makedirs(batch_dir, exist_ok=True)
print(f"Output folder: {batch_dir}")

# Enable motion blur
stage = omni.usd.get_context().get_stage()
render_settings = stage.GetPrimAtPath("/Render/OmniverseGlobalRenderSettings")
render_settings.GetAttribute("disableMotionBlur").Set(False)
render_settings.GetAttribute("instantaneousShutter").Set(False)
print("Motion blur enabled")

with rep.new_layer():

    camera = rep.create.camera(
        position=(0, 0, 500),
        look_at=(0, 0, 0)
    )

    render_product = rep.create.render_product(camera, resolution=(640, 480))

    landing_pad = rep.get.prims("/World/LandingPad")
    with landing_pad:
        rep.modify.semantics([("class", "landing_pad")])

    sun_light  = rep.get.prims("/World/SunLight")
    dome_light = rep.get.prims("/World/DomeLight")

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=batch_dir,
        rgb=True,
        bounding_box_2d_tight=True,
    )
    writer.attach([render_product])

    with rep.trigger.on_frame(num_frames=FRAMES_PER_BATCH, rt_subframes=4):

        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (-400, -400, 300),
                    ( 400,  400, 600),
                ),
                look_at=rep.distribution.uniform(
                    (-80, -80, 0),
                    ( 80,  80, 0),
                )
            )

        with sun_light:
            rep.modify.attribute(
                "inputs:intensity",
                rep.distribution.uniform(1000.0, 15000.0)
            )
            rep.modify.attribute(
                "inputs:color",
                rep.distribution.uniform(
                    (1.0, 0.8, 0.6),
                    (1.0, 1.0, 1.0),
                )
            )
            rep.modify.attribute(
                "xformOp:rotateXYZ",
                rep.distribution.uniform(
                    (-85, 0,   0),
                    (-5,  360, 0)
                )
            )

        with dome_light:
            rep.modify.attribute(
                "inputs:intensity",
                rep.distribution.uniform(250.0, 500.0)
            )
            rep.modify.attribute(
                "inputs:color",
                rep.distribution.uniform(
                    (0.7, 0.8, 1.0),
                    (1.0, 1.0, 1.0),
                )
            )

rep.orchestrator.preview()
time.sleep(15)
writer.attach([render_product])
rep.orchestrator.run()

time.sleep(120)
print(f"✓ Batch {BATCH_INDEX:04d} done — {FRAMES_PER_BATCH} frames saved to {batch_dir}")
print(f"  Next: set BATCH_INDEX = {BATCH_INDEX + 1} and run in fresh session")