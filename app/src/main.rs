use askama::Template;
use axum::{
    Json, Router,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::json;
use tch::{Device, Tensor, nn};
use tokio::net::TcpListener;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let app = Router::new()
        .route("/", get(home))
        .route("/guess", post(guess));

    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}

#[derive(Template)]
#[template(path = "index.html")]
struct MyTemplate;

async fn home() -> Html<String> {
    let template = MyTemplate {}.render().unwrap();

    Html(template)
}

#[derive(Deserialize)]
struct Payload {
    grid: Vec<f32>,
}

async fn guess(Json(payload): Json<Payload>) -> impl IntoResponse {
    let device = Device::cuda_if_available();

    let mut vs = nn::VarStore::new(device);
    let root = &vs.root();

    let model = nn::seq()
        // conv 1
        .add(nn::conv2d(root, 1, 16, 3, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(2, 2, 0, 1, false))
        // conv 2
        .add(nn::conv2d(root, 16, 32, 3, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(2, 2, 0, 1, false))
        // flatten
        .add_fn(|xs| xs.flatten(1, -1))
        // linear 1
        .add(nn::linear(root, 32 * 5 * 5, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        // output layer
        .add(nn::linear(root, 64, 10, Default::default()));

    vs.load("models/model.ot").unwrap();

    let guess = Tensor::from_slice(&payload.grid)
        .reshape([1, 1, 28, 28])
        .apply(&model)
        .argmax(1, false)
        .int64_value(&[]);

    (
        StatusCode::OK,
        Json(json!({
            "guess": guess
        })),
    )
}
