import graphene as gr


class Update(gr.ObjectType):
    pk = gr.ID(required=True)
    id = gr.UUID(required=True)
    algorithm = gr.String(required=True)
    ensemble_reference_pk = gr.Int()
    ensemble_result_pk = gr.Int()
    ensemble_reference = gr.Field(
        "ert_shared.dark_storage.graphql.ensembles.CreateEnsemble"
    )
    ensemble_result = gr.Field(
        "ert_shared.dark_storage.graphql.ensembles.CreateEnsemble"
    )

    def resolve_pk(*args):
        pass
