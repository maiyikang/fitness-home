# ==========================================================
# Validation
# ==========================================================


def is_valid_sample(
    sample: Dict[str, Any],
) -> bool:

    output = normalize_text(
        sample["output"]
    )

    metadata = sample["metadata"]

    restaurant = normalize_text(
        metadata["restaurant"]
    )

    menu = normalize_text(
        metadata["menu"]
    )

    if restaurant == "":
        return False

    if menu == "":
        return False

    if restaurant not in output:
        return False

    if menu not in output:
        return False

    if "Recommended Reason:" not in output:
        return False

    return True


# ==========================================================
# Metadata
# ==========================================================


def save_generation_metadata(

    total_candidates: int,

    total_samples: int,

    failed_samples: int,

) -> None:

    metadata = {

        "teacher_model": MODEL_NAME,

        "dataset_version": DATASET_VERSION,

        "candidate_count": total_candidates,

        "generated_samples": total_samples,

        "failed_samples": failed_samples,

        "queries_per_menu": QUERIES_PER_MENU,

        "max_menu_candidates": MAX_MENU_CANDIDATES,

        "max_samples": MAX_SAMPLES,

    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ==========================================================
# Dataset Generator
# ==========================================================


def generate_dataset() -> None:

    random.seed(RANDOM_SEED)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = load_candidates()

    print("=" * 60)
    print("Fitness Home Dataset Generator")
    print("=" * 60)
    print(f"Teacher Model : {MODEL_NAME}")
    print(f"Candidates    : {len(candidates)}")
    print(f"Max Samples   : {MAX_SAMPLES}")
    print("=" * 60)

    generated_samples = 0

    failed_samples = 0

    seen: Set[str] = set()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for item in candidates:

            if generated_samples >= MAX_SAMPLES:
                break

            request_infos = build_user_requests(
                item
            )

            for request_info in request_infos:

                if generated_samples >= MAX_SAMPLES:
                    break

                unique_key = (

                    f"{item['restaurant_id']}_"

                    f"{item['menu_id']}_"

                    f"{request_info['structured_input']}"

                )

                if unique_key in seen:
                    continue

                seen.add(unique_key)

                print(
                    f"[{generated_samples + 1}/{MAX_SAMPLES}] "
                    f"{item['restaurant_name']} "
                    f"-> "
                    f"{item['menu_name']}"
                )

                prompt = build_teacher_prompt(
                    item,
                    request_info["structured_input"],
                )

                try:

                    explanation = generate_explanation(
                        prompt
                    )

                    sample = build_sample(
                        item,
                        request_info,
                        explanation,
                    )

                    if not is_valid_sample(sample):

                        failed_samples += 1

                        continue

                    file.write(
                        json.dumps(
                            sample,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    generated_samples += 1

                    if REQUEST_DELAY_SECONDS > 0:

                        time.sleep(
                            REQUEST_DELAY_SECONDS
                        )

                except Exception as error:

                    failed_samples += 1

                    print(
                        f"Generation failed: {error}"
                    )

                    continue

    save_generation_metadata(

        len(candidates),

        generated_samples,

        failed_samples,

    )

    print()

    print("=" * 60)

    print("Dataset Generation Completed")

    print("=" * 60)

    print(f"Generated Samples : {generated_samples}")

    print(f"Failed Samples    : {failed_samples}")

    print(f"Output            : {OUTPUT_FILE}")

    print(f"Metadata          : {METADATA_FILE}")

    print("=" * 60)


# ==========================================================
# Main
# ==========================================================


def main() -> None:

    generate_dataset()


if __name__ == "__main__":

    main()