fn main() {

	//Saving Target information to "TARGET" environment variable
    println!(
        "cargo:rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap()
    );
}