# flows/build_features.py

from prefect import flow, get_run_logger, task

from customer_ai.features import (
    build_churn_features,
    build_cltv_summary,
    build_order_features,
    build_sales_fact_with_churn,
    build_segment_features,
    build_segment_features_extended,
)


@task
def wrap(builder_fn):
    """Run one of your customer_ai.features builders as a Prefect task."""
    path = builder_fn()
    get_run_logger().info(f"✅ Wrote {path.name}")
    return path


@flow(name="feature_build_flow")
def feature_build_flow():
    # run them in sequence (or .submit() in parallel if you prefer)
    wrap(build_sales_fact_with_churn)
    wrap(build_order_features)
    wrap(build_churn_features)
    wrap(build_segment_features)
    wrap(build_segment_features_extended)
    wrap(build_cltv_summary)
    get_run_logger().info("🎉 All feature tables built")


if __name__ == "__main__":
    feature_build_flow()
